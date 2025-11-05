import os
import time
import torch
import torch.nn as nn
import dgl
import dgl.nn.pytorch as dglnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
import sys

# 添加当前目录到系统路径，确保可以正确导入models模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from dgl.dataloading import EdgeDataLoader, as_edge_prediction_sampler, NeighborSampler

from models.gcn import GCN
from datasets.data_generator import GraphGenerator

# ==========================================================
# 1️⃣ 设置训练参数
# ==========================================================
# 分布式训练参数


os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "4"
os.environ["LOCAL_RANK"] = "0"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 4))
rank = int(os.environ.get("RANK", 0))

# 训练参数
num_epochs = 100
lr = 0.001
batch_size = 1024
num_workers = 0

# 图划分参数
graph_dir = "datasets/graph_parts"
num_parts = 3

# 设备设置
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

# ==========================================================
# 2️⃣ 初始化分布式训练环境
# ==========================================================
print(f"🚀 初始化分布式训练环境...")
print(f"   Rank: {rank} | Local Rank: {local_rank} | World Size: {world_size}")
# 如果环境变量RANK未定义，则默认设为0

# dist.init_process_group(backend="nccl")

# ==========================================================
# 3️⃣ 生成图数据并划分
# ==========================================================
print(f"📊 生成图数据并划分...")

# 检查图划分文件是否存在，如果不存在则生成
if not os.path.exists(graph_dir):
    print(f"🔧 生成新的图数据并划分...")
    os.makedirs(graph_dir, exist_ok=True)
    
    # 生成图数据
    gen = GraphGenerator()
    G_nx = gen.generate_nx_graph(kind='ER', n_nodes=2000, p=0.01)
    g = gen.nx_to_dgl(G_nx)
    gen.add_node_labels(g)
    
    # 划分为子图
    gen.partition_graph_for_node_classification(g, num_parts=num_parts, method='metis', output_dir=graph_dir)
    gen.partition_graph_for_node_classification(g, num_parts=num_parts, method='random', output_dir=graph_dir)
    
    print(f"✅ 图数据生成和划分完成")

# 加载分区图
part_config = os.path.join(graph_dir, "synthetic_lp_graph.json")
if not os.path.exists(part_config):
    print(f"❌ 图划分配置文件 {part_config} 不存在")
    exit(1)



# # 加载分区图
# dist_g = dgl.distributed.DistGraph(
#     graph_name="synthetic_lp_graph",
#     part_config=part_config
# )
gen = GraphGenerator()
loader, _ = gen.get_dataloader_for_node_classification(pid=0, partition_method='metis', partition_dir=graph_dir)

# print(f"✅ 图数据加载完成:")
# print(f"   图名称: {dist_g.graph_name}")
# print(f"   分区数量: {num_parts}")
# print(f"   节点总数: {dist_g.number_of_nodes()}")
# print(f"   边总数: {dist_g.number_of_edges()}")


# ==========================================================
# 5️⃣ 构建采样器与数据加载器
# ==========================================================
# # 采用邻居采样策略 (2-hop)
# sampler = as_edge_prediction_sampler(
#     NeighborSampler([10, 10]),
#     negative_sampler=dgl.dataloading.negative_sampler.Uniform(1)
# )

# # 这里我们简单使用所有边索引
# edge_ids = torch.arange(g.num_edges())




# EdgeDataLoader 支持多进程分布式加载
# dataloader = EdgeDataLoader(
#     dist_g,
#     edge_ids,
#     sampler,
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=0
# )

# ==========================================================
# 6️⃣ 初始化 TensorBoard 记录器
# ==========================================================
writer = SummaryWriter(log_dir='runs/link_prediction_experiment')

# ==========================================================
# 7️⃣ 初始化 GNN 模型（链路预测任务）
# ==========================================================
# 基于生成的图数据：输入维度128，输出维度1（边存在概率）
input_dim = 128  # 节点特征维度
hidden_dim = 64  # 隐藏层维度
output_dim = 1   # 直接输出边存在概率

# 初始化GCN模型，直接输出边预测概率
gnn = GCN(input_dim, hidden_dim, output_dim, dropout=0.5)
gnn = gnn.to(device)

# 使用 DistributedDataParallel 包装模型
gnn = torch.nn.parallel.DistributedDataParallel(gnn, device_ids=[local_rank])

# 优化器
optimizer = torch.optim.Adam(gnn.parameters(), lr=lr)

print(f"🚀 模型初始化完成:")
print(f"   输入维度: {input_dim}")
print(f"   隐藏维度: {hidden_dim}")
print(f"   输出维度: {output_dim}")
print(f"   总参数量: {sum(p.numel() for p in gnn.parameters()):,}")
print(f"   分布式训练: 是 | GPU数量: {torch.cuda.device_count()}")


# ==========================================================
# 8️⃣ 训练循环
# ==========================================================
print("🎯 开始训练链路预测模型 ...")

# 初始化带宽监控变量
prev_comm_time = 0

gnn.train()
for epoch in range(num_epochs):
    total_loss = 0
    num_batches = 0
    epoch_comm_time = 0
    epoch_forward_time = 0
    epoch_backward_time = 0
    
    for input_nodes, pair_graph, blocks in dataloader:
        # 通信时间监控开始
        comm_start_time = time.time()
        
        # 获取节点特征
        input_features = dist_g.ndata['feats'][input_nodes].to(device)
        
        # 通信时间监控结束
        comm_end_time = time.time()
        comm_time = comm_end_time - comm_start_time
        epoch_comm_time += comm_time
        
        # 前向传播时间监控
        forward_start_time = time.time()
        node_embeddings = gnn(blocks[0], input_features)
        forward_end_time = time.time()
        forward_time = forward_end_time - forward_start_time
        epoch_forward_time += forward_time
        
        # 获取边预测结果 - 拼接两个节点采样后的特征作为输入向量
        src_features = blocks[0].srcdata['feats'][pair_graph.edges()[0]]
        dst_features = blocks[0].srcdata['feats'][pair_graph.edges()[1]]
        
        # 拼接源节点和目标节点的特征
        combined_features = torch.cat([src_features, dst_features], dim=1)
        
        # 直接使用GCN计算边得分（输出维度为1）
        edge_scores = gnn(pair_graph, combined_features).squeeze(1)
        
        # 获取标签
        edge_labels = pair_graph.edata['label'].to(device).float()
        
        # 计算二元交叉熵损失
        loss = nn.functional.binary_cross_entropy_with_logits(
            edge_scores, edge_labels
        )
        
        # 反向传播时间监控
        backward_start_time = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        backward_end_time = time.time()
        backward_time = backward_end_time - backward_start_time
        epoch_backward_time += backward_time
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_comm_time = epoch_comm_time / num_batches if num_batches > 0 else 0
    avg_forward_time = epoch_forward_time / num_batches if num_batches > 0 else 0
    avg_backward_time = epoch_backward_time / num_batches if num_batches > 0 else 0
    
    # 计算带宽波动（基于通信时间的变化）
    if epoch > 0:
        bandwidth_variation = abs(avg_comm_time - prev_comm_time) / prev_comm_time if prev_comm_time > 0 else 0
        writer.add_scalar('Bandwidth/variation', bandwidth_variation, epoch)
    prev_comm_time = avg_comm_time
    
    # 记录到 TensorBoard
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('Time/communication', avg_comm_time, epoch)
    writer.add_scalar('Time/forward', avg_forward_time, epoch)
    writer.add_scalar('Time/backward', avg_backward_time, epoch)
    writer.add_scalar('Time/total', avg_comm_time + avg_forward_time + avg_backward_time, epoch)
    
    print(f"📊 Epoch [{epoch+1}/{num_epochs}] | 平均损失: {avg_loss:.4f}")
    print(f"   ⏱️  通信时间: {avg_comm_time:.4f}s | 前向时间: {avg_forward_time:.4f}s | 反向时间: {avg_backward_time:.4f}s")


# ==========================================================
# 9️⃣ 清理资源
# ==========================================================
print("🧹 清理分布式资源 ...")

# 关闭 TensorBoard 记录器
writer.close()

# 清理分布式环境
dist.destroy_process_group()
print("✅ 训练完成！")
print("📈 训练日志已保存到 'runs/link_prediction_experiment'，使用 'tensorboard --logdir=runs' 查看")
