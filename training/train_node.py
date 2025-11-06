import os
import time
import torch
import torch.nn as nn
import dgl
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import sys
import psutil
import GPUtil
import torch.multiprocessing as mp

# ==========================================================
# 添加当前目录到系统路径
# ==========================================================
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.gcn import GCN
from datasets.data_generator import GraphGenerator
import datetime


# ==========================================================
# 1️⃣ 初始化分布式训练环境
# ==========================================================
def setup_distributed(rank, world_size):
    """初始化分布式环境"""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", 
                            rank=rank, 
                            world_size=world_size)
    device = torch.device(f"cuda:{rank}")
    
    print(f"🚀 Rank {rank}: 初始化分布式训练环境")
    return device


def train_fn(rank, world_size, graph_dir, num_epochs=10, lr=0.001):
    """分布式训练函数，由mp.spawn调用"""
    device = None
    try:
        device = setup_distributed(rank, world_size)
        local_rank = rank
        print(f"Rank {rank} 启动训练进程")
        # 训练函数调用
        train(rank, local_rank, world_size, device, graph_dir=graph_dir, num_epochs=num_epochs, lr=lr)
    except Exception as e:
        print(f"Rank {rank} 训练过程中出现异常: {e}")
    finally:
        # 只在进程组成功初始化后才销毁
        if dist.is_initialized():
            dist.destroy_process_group()


# ==========================================================
# 2️⃣ 图生成与划分
# ==========================================================
def prepare_graph(graph_dir="datasets/graph_parts", num_parts=3, nodes=20):
    """生成或加载划分好的图"""
    if not os.path.exists(graph_dir):
        print(f"🔧 生成新的图数据并划分...")
        os.makedirs(graph_dir, exist_ok=True)
        gen = GraphGenerator()
        G_nx = gen.generate_nx_graph(kind='ER', n_nodes=nodes, p=0.01,)
        g = gen.nx_to_dgl(G_nx)
        gen.add_node_labels(g)

        # 同时生成 metis 和 random 划分
        gen.partition_graph(g, num_parts=num_parts, method='metis', output_dir=graph_dir)

        print(f"✅ 图数据生成和划分完成")

    return graph_dir


# ==========================================================
# 3️⃣ 训练函数
# ==========================================================
def train(rank, local_rank, world_size, device, graph_dir, num_epochs=20, lr=0.001, partition_method="metis"):
    
    torch.manual_seed(0)

    # 初始化 TensorBoard，仅 rank 0 写日志
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=f"runs/node_classification_{timestamp}_rank{rank}") if rank == 0 else None

    # 数据加载器初始化 - 每个rank加载对应的子图分区
    gen = GraphGenerator()
    train_loader, test_loader, subg = gen.get_dataloader_for_node_classification(
        pid=rank,
        partition_method=partition_method,
        batch_size=32,
        train_ratio=0.8,
        num_workers=0,
        device=device,
        sampler_fanouts=[10, 5],
        partition_dir=graph_dir
    )

    if rank == 0:
        print(f"📊 Rank {rank} 加载完成 dataloader（子图 {rank}）")

    # 模型初始化
    input_dim = 4
    hidden_dim = 64
    num_classes = subg.ndata['labels'].max().item() + 1
    gnn = GCN(input_dim, hidden_dim, num_classes, dropout=0.0).to(device)
    gnn = DDP(gnn, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=lr)

    if rank == 0:
        print(f"🚀 模型初始化完成: {sum(p.numel() for p in gnn.parameters()):,} 参数")

    # ============ 开始训练 ============ 
    gnn.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        for input_nodes, output_nodes, blocks in train_loader:
            # 1️⃣ 获取节点特征和标签
            feats = blocks[0].srcdata["feat"].to(device)
            labels = blocks[-1].dstdata["labels"].to(device)  # 最后一层 block 的 dst 节点标签
            
            # 调试：检查维度
            print(f"Debug - input_nodes length: {len(input_nodes)}")
            print(f"Debug - output_nodes length: {len(output_nodes)}")
            print(f"Debug - blocks[0] src nodes: {blocks[0].num_src_nodes()}, dst nodes: {blocks[0].num_dst_nodes()}")
            print(f"Debug - blocks[-1] src nodes: {blocks[-1].num_src_nodes()}, dst nodes: {blocks[-1].num_dst_nodes()}")
            print(f"Debug - feats shape: {feats.shape}")
            print(f"Debug - labels shape: {labels.shape}")

            # 2️⃣ 前向传播
            logits = gnn(blocks, feats)

            print(f"Debug - logits shape: {logits.shape}")

            # 3️⃣ 计算交叉熵损失
            loss = nn.functional.cross_entropy(logits, labels)

            # 4️⃣ 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 5️⃣ 计算准确率
            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean().item()

            total_loss += loss.item()
            total_accuracy += acc
            num_batches += 1

            if num_batches % 10 == 0:
                print(f"Rank {rank} batch {num_batches}, loss: {loss.item():.4f}, acc: {acc:.4f}")

        # 同步平均 loss 和 accuracy
        avg_loss_tensor = torch.tensor(total_loss / max(num_batches, 1), device=device)
        avg_acc_tensor = torch.tensor(total_accuracy / max(num_batches, 1), device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(avg_acc_tensor, op=dist.ReduceOp.AVG)
        avg_loss = avg_loss_tensor.item()
        avg_accuracy = avg_acc_tensor.item()

        if writer and rank == 0:
            writer.add_scalar("Loss/train", avg_loss, epoch)
            writer.add_scalar("Accuracy/train", avg_accuracy, epoch)

        print(f"Rank {rank} [Epoch {epoch+1}/{num_epochs}] 平均损失: {avg_loss:.4f}, 准确度: {avg_accuracy:.4f}")

    dist.destroy_process_group()
    if writer:
        writer.close()
    if rank == 0:
        print("✅ Node classification 训练完成！")



# ==========================================================
# 4️⃣ 主入口
# ==========================================================
if __name__ == "__main__":
    graph_dir = prepare_graph(graph_dir="datasets/node_cls_small_twolayer", num_parts=3, nodes=200)
    world_size = 3
    device = None
    # 使用mp.spawn启动分布式训练
    mp.spawn(
        train_fn,
        args=(world_size, graph_dir),
        nprocs=world_size,
        join=True
    )
