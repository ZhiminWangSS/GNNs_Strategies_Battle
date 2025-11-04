import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from models.gcn import GCN
from utils.graph_partition import partition_dgl_graph

import dgl
import dgl.nn.pytorch as dglnn
from dgl.dataloading import EdgeDataLoader, NeighborSampler, as_edge_prediction_sampler
from dgl.distributed import DistGraph, partition_graph, initialize, finalize


graph_name = "synthetic_lp_graph"
graph_dir = "graph_partitions"
graph_bin = "synthetic_lp_graph.bin"
edge_file = "link_prediction_edges.pt"
num_parts = 4
partition_method = "metis"  # 可选值：'metis' / 'random' / 'non_uniform'
num_epochs = 5
batch_size = 1024
lr = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================================
# 2️⃣ 读取图数据
# ==========================================================
print("📦 加载图数据中 ...")
g_list, _ = dgl.load_graphs(graph_bin)
g = g_list[0]

# 加载边样本（正负样本）
edge_data = torch.load(edge_file)
edges = edge_data["edges"]
labels = edge_data["labels"]

print(f"图节点数: {g.num_nodes()} | 边数: {g.num_edges()}")
print(f"链路预测样本数量: {len(edges)}")



# ==========================================================
# 3️⃣ 图划分（根据选择）
# ==========================================================
os.makedirs(graph_dir, exist_ok=True)

partition_dgl_graph(g, graph_name, graph_dir, partition_method, num_parts)

# 仅在划分文件不存在时执行划分
if not os.path.exists(os.path.join(graph_dir, graph_name + ".json")):
    partition_dgl_graph(g, graph_name, graph_dir, partition_method, num_parts)


# ==========================================================
# 4️⃣ 初始化分布式图（单机多卡环境）
# ==========================================================
print("🚀 初始化分布式图 ...")
initialize("graph_partitions")  # 初始化 DGL 分布式图引擎
dist_g = DistGraph(graph_name, part_config=os.path.join(graph_dir, graph_name + ".json"))
print("分布式图加载成功 ✅")


# ==========================================================
# 5️⃣ 构建采样器与数据加载器
# ==========================================================
# 采用邻居采样策略 (2-hop)
sampler = as_edge_prediction_sampler(
    NeighborSampler([10, 10]),
    negative_sampler=dgl.dataloading.negative_sampler.Uniform(1)
)

# 这里我们简单使用所有边索引
edge_ids = torch.arange(g.num_edges())

# EdgeDataLoader 支持多进程分布式加载
dataloader = EdgeDataLoader(
    dist_g,
    edge_ids,
    sampler,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

# ==========================================================
# 6️⃣ 初始化 TensorBoard 记录器
# ==========================================================
writer = SummaryWriter(log_dir='runs/link_prediction_experiment')

# ==========================================================
# 7️⃣ 初始化 GNN 模型（链路预测任务）
# ==========================================================
# 基于 OGBL-ddi 数据集：输入维度128，输出维度1（边存在概率）
input_dim = 128  # 节点特征维度
hidden_dim = 64  # 隐藏层维度
output_dim = 1   # 链路预测输出维度（边存在概率）

gnn = GCN(input_dim, hidden_dim, output_dim, dropout=0.5)
gnn = gnn.to(device)

# 优化器
optimizer = torch.optim.Adam(gnn.parameters(), lr=lr)

print(f"🚀 GNN 模型初始化完成:")
print(f"   输入维度: {input_dim}")
print(f"   隐藏维度: {hidden_dim}")
print(f"   输出维度: {output_dim}")
print(f"   总参数量: {sum(p.numel() for p in gnn.parameters()):,}")


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
        input_features = dist_g.ndata['feat'][input_nodes].to(device)
        
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
        
        # 获取边预测结果
        src_embeddings = node_embeddings[pair_graph.edges()[0]]
        dst_embeddings = node_embeddings[pair_graph.edges()[1]]
        
        # 计算边得分（点积相似度）
        edge_scores = torch.sum(src_embeddings * dst_embeddings, dim=1)
        
        # 获取标签
        edge_labels = pair_graph.edata['label'].to(device).float()
        
        # 计算损失（二元交叉熵）
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
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

finalize()
print("✅ 训练完成！")
print("📈 训练日志已保存到 'runs/link_prediction_experiment'，使用 'tensorboard --logdir=runs' 查看")
