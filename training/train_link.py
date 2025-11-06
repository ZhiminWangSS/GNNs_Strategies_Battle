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
    """
    初始化分布式训练环境
    
    关键参数设置:
    - MASTER_ADDR: 主节点地址，通常为localhost
    - MASTER_PORT: 通信端口
    - backend: 通信后端，GPU推荐使用nccl
    """
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





# ==========================================================
# 2️⃣ 图生成与划分
# ==========================================================
def prepare_graph(graph_dir="datasets/graph_parts", num_parts=3, nodes=20):
    """
    生成或加载划分好的图数据
    
    关键参数设置:
    - graph_dir: 图数据存储目录
    - num_parts: 图划分份数，对应分布式进程数
    - nodes: 图节点数量
    - p: ER图生成概率
    """
    if not os.path.exists(graph_dir):
        print(f"🔧 生成新的图数据并划分...")
        os.makedirs(graph_dir, exist_ok=True)
        gen = GraphGenerator()
        G_nx = gen.generate_nx_graph(kind='ER', n_nodes=nodes, p=0.01)
        g = gen.nx_to_dgl(G_nx)
        gen.add_node_labels(g)

        # 使用metis算法进行图划分
        gen.partition_graph(g, num_parts=num_parts, method='metis', output_dir=graph_dir)

        print(f"✅ 图数据生成和划分完成")

    return graph_dir


def train_fn(rank, world_size, graph_dir, num_epochs=100, lr=0.001):
    """
    分布式训练函数，由mp.spawn调用
    
    关键参数设置:
    - num_epochs: 训练轮数
    - lr: 学习率
    """
    device = setup_distributed(rank, world_size)
    local_rank = rank
    print(f"Rank {rank} 启动训练进程")
    # 调用主训练函数
    train(rank, local_rank, world_size, device, graph_dir=graph_dir, num_epochs=num_epochs, lr=lr)
    
    dist.destroy_process_group()

# ==========================================================
# 3️⃣ 训练函数
# ==========================================================
def train(rank, local_rank, world_size, device, graph_dir, num_epochs=20, lr=0.001, partition_method="metis"):
    """
    链路预测训练主函数
    
    关键参数设置:
    - num_epochs: 训练轮数
    - lr: 学习率
    - partition_method: 图划分方法
    - batch_size: 批次大小
    - train_ratio: 训练集比例
    - sampler_fanouts: 邻居采样层数配置
    - input_dim/hidden_dim/output_dim: 模型维度配置
    """
    torch.manual_seed(0)

    # 初始化 TensorBoard，仅 rank 0 写日志
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=f"runs/link_prediction_{timestamp}_rank{rank}") if rank == 0 else None

    # 数据加载器初始化 - 每个rank加载对应的子图分区
    gen = GraphGenerator()
    train_loader, test_loader, subg = gen.get_dataloader_for_link_prediction(
        pid=rank,  # 使用rank作为分区ID
        partition_method=partition_method,
        batch_size=32,  # 关键参数: 批次大小
        train_ratio=0.8,  # 关键参数: 训练集比例
        num_workers=0,
        device=device,
        sampler_fanouts=[10, 10, 5],  # 关键参数: 邻居采样配置
        partition_dir=graph_dir
    )
    print(f"Batch size: {train_loader.batch_size}")
    num_batches = len(train_loader)
    print(f"每个 epoch 需要迭代 {num_batches} 次")
    if rank == 0:
        print(f"📊 Rank {rank} 加载完成 dataloader（子图 {rank}）")
    subg = subg.to(device)
    # 模型初始化
    input_dim = 4   # 关键参数: 输入特征维度
    hidden_dim = 64  # 关键参数: 隐藏层维度
    output_dim = 1   # 关键参数: 输出维度(链路预测得分)
    
    gnn = GCN(input_dim, hidden_dim, output_dim, dropout=0.0).to(device)
    gnn = DDP(gnn, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=lr)

    if rank == 0:
        print(f"🚀 模型初始化完成: {sum(p.numel() for p in gnn.parameters()):,} 参数")

    # ============ 开始训练 ============
    gnn.train()
    prev_comm_time = 0

    # 同步各进程的批次数量，确保分布式训练同步
    local_num_batches = len(train_loader)
    num_batches_tensor = torch.tensor(local_num_batches, device=device)
    all_num_batches = [torch.zeros_like(num_batches_tensor) for _ in range(world_size)]
    dist.all_gather(all_num_batches, num_batches_tensor)
    max_num_batches = max([x.item() for x in all_num_batches])
    
    # ============ 训练循环 ============
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        epoch_comm_time = 0
        epoch_forward_time = 0
        epoch_backward_time = 0
        
        iter = 1
        if iter <= local_num_batches:
            for input_nodes, pos_pair_graph, neg_pair_graph, blocks in train_loader:
                
                comm_start_time = time.time()

                # 1️⃣ 取出节点特征（输入层源节点）
                feats = subg.ndata['feat'][input_nodes].to(device)
                # feats = blocks[0].srcdata["feat"].to(device)

                # 2️⃣ 使用 blocks 做 GCN 前向编码（message passing）
                # gnn 的 forward 需要 (blocks, feats)
                node_emb = gnn(blocks, feats)   # 输出的是目标节点的embedding（最后一层block的dst节点）

                comm_end_time = time.time()
                epoch_comm_time += (comm_end_time - comm_start_time)
                
                # 3️⃣ 从正样本图中取出边两端节点的 embedding
                pos_src, pos_dst = pos_pair_graph.edges()
                pos_src_emb = node_emb[pos_src]
                pos_dst_emb = node_emb[pos_dst]
                # 点乘得分
                pos_score = (pos_src_emb * pos_dst_emb).sum(dim=1)

                # 4️⃣ 从负样本图中取出边两端节点的 embedding
                neg_src, neg_dst = neg_pair_graph.edges()
                neg_src_emb = node_emb[neg_src]
                neg_dst_emb = node_emb[neg_dst]
                neg_score = (neg_src_emb * neg_dst_emb).sum(dim=1)
                
                # 5️⃣ 计算链路预测损失（负样本标签为 0，正样本为 1）
                scores = torch.cat([pos_score, neg_score], dim=0)
                labels = torch.cat([
                    torch.ones_like(pos_score),
                    torch.zeros_like(neg_score)
                ]).to(device)

                forward_start = time.time()
                loss = nn.functional.binary_cross_entropy_with_logits(scores, labels)
                forward_end = time.time()
                epoch_forward_time += (forward_end - forward_start)
                
                # 6️⃣ 反向传播与优化
                backward_start = time.time()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                backward_end = time.time()
                epoch_backward_time += (backward_end - backward_start)

                # 计算准确度
                predictions = (torch.sigmoid(scores) > 0.5).float()
                accuracy = (predictions == labels).float().mean().item()
                
                # 获取内存使用情况
                process = psutil.Process()
                memory_usage = process.memory_info().rss / 1024 / 1024  # MB
                
                # 获取GPU内存使用情况
                gpu_memory_usage = 0
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_memory_usage = gpus[local_rank].memoryUsed
                except:
                    pass

                total_loss += loss.item()
                total_accuracy += accuracy
                num_batches += 1
                
                if num_batches % 10 == 0:
                    print(f"Rank {rank} batch {num_batches} 内训练中, loss: {loss.item():.4f}, acc: {accuracy:.4f}")
                iter += 1
        
        # 7️⃣ 同步其他进程的批次训练（确保分布式训练同步）
        if iter > local_num_batches:
            for _ in range(max_num_batches - local_num_batches):
                dist.all_reduce(torch.zeros(1, device=device))

        # ============ 计算训练指标 ============
        # 计算平均损失和准确度
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches

        # 同步所有进程的损失和准确度
        avg_loss_tensor = torch.tensor(avg_loss, device=device)
        avg_accuracy_tensor = torch.tensor(avg_accuracy, device=device)
        
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(avg_accuracy_tensor, op=dist.ReduceOp.SUM)
        
        avg_loss_tensor /= world_size
        avg_accuracy_tensor /= world_size

        # 记录训练时间
        epoch_time = time.time() - epoch_start_time

        # ============ 日志记录 ============
        # 记录到 TensorBoard（仅在 rank 0 进程记录）
        if rank == 0:
            writer.add_scalar("Loss/train", avg_loss_tensor.item(), epoch)
            writer.add_scalar("Accuracy/train", avg_accuracy_tensor.item(), epoch)
            writer.add_scalar("Time/epoch", epoch_time, epoch)
            writer.add_scalar("Time/communication", epoch_comm_time, epoch)
            writer.add_scalar("Time/forward", epoch_forward_time, epoch)
            writer.add_scalar("Time/backward", epoch_backward_time, epoch)
            writer.add_scalar("Memory/usage", memory_usage, epoch)

        # 打印训练信息
        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, "
                  f"Loss: {avg_loss_tensor.item():.4f}, "
                  f"Accuracy: {avg_accuracy_tensor.item():.4f}, "
                  f"Time: {epoch_time:.2f}s")

        # # ============ 测试评估 ============
        # # 每 10 个 epoch 进行一次测试评估
        # if (epoch + 1) % 10 == 0:
        #     test_accuracy = evaluate(gnn, test_loader, device, rank, world_size)
        #     if rank == 0:
        #         writer.add_scalar("Accuracy/test", test_accuracy, epoch)
        #         print(f"Test Accuracy at epoch {epoch+1}: {test_accuracy:.4f}")

    # ============ 模型保存 ============
    # 保存模型（仅在 rank 0 进程保存）
    if rank == 0:
        torch.save(gnn.state_dict(), f"link_prediction_model_rank{rank}.pth")
        print(f"Model saved as link_prediction_model_rank{rank}.pth")

    # ============ 最终测试评估 ============
    # 在所有训练结束后进行完整的测试评估
    final_test_accuracy = evaluate(gnn, test_loader, device, rank, world_size)
    
    # 收集所有进程的测试准确度
    test_acc_tensor = torch.tensor(final_test_accuracy, device=device)
    all_test_acc = [torch.zeros_like(test_acc_tensor) for _ in range(world_size)]
    dist.all_gather(all_test_acc, test_acc_tensor)
    
    # 计算平均测试准确度
    avg_test_accuracy = sum([acc.item() for acc in all_test_acc]) / world_size
    
    if rank == 0:
        print(f"\n📊 最终测试结果:")
        print(f"各进程测试准确度: {[acc.item() for acc in all_test_acc]}")
        print(f"平均测试准确度: {avg_test_accuracy:.4f}")
        writer.add_scalar("Accuracy/final_test", avg_test_accuracy, num_epochs)

    # 关闭 TensorBoard writer
    if rank == 0:
        writer.close()

    print(f"Rank {rank} training completed.")


# ==========================================================
# 5️⃣ 评估函数
# ==========================================================
def evaluate(model, test_loader, device, rank, world_size):
    """
    模型评估函数
    
    关键参数设置:
    - model: 待评估的GNN模型
    - test_loader: 测试数据加载器
    - device: 计算设备
    - rank: 当前进程排名
    - world_size: 进程总数
    """
    model.eval()
    total_accuracy = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for input_nodes, pos_pair_graph, neg_pair_graph, blocks in test_loader:
            # 1️⃣ 获取节点特征
            feats = blocks[0].srcdata["feat"].to(device)
            
            # 2️⃣ GCN前向传播获取节点嵌入
            node_emb = model(blocks, feats)
            
            # 3️⃣ 计算正样本和负样本的预测得分
            pos_src, pos_dst = pos_pair_graph.edges()
            pos_src_emb = node_emb[pos_src]
            pos_dst_emb = node_emb[pos_dst]
            pos_score = (pos_src_emb * pos_dst_emb).sum(dim=1)
            
            neg_src, neg_dst = neg_pair_graph.edges()
            neg_src_emb = node_emb[neg_src]
            neg_dst_emb = node_emb[neg_dst]
            neg_score = (neg_src_emb * neg_dst_emb).sum(dim=1)
            
            # 4️⃣ 合并正负样本得分和标签
            scores = torch.cat([pos_score, neg_score], dim=0)
            labels = torch.cat([
                torch.ones_like(pos_score),
                torch.zeros_like(neg_score)
            ]).to(device)
            
            # 5️⃣ 计算准确度
            predictions = (torch.sigmoid(scores) > 0.5).float()
            accuracy = (predictions == labels).float().mean().item()
            
            total_accuracy += accuracy
            num_batches += 1
    
    # 6️⃣ 同步所有进程的准确度
    avg_accuracy = total_accuracy / num_batches if num_batches > 0 else 0.0
    avg_accuracy_tensor = torch.tensor(avg_accuracy, device=device)
    dist.all_reduce(avg_accuracy_tensor, op=dist.ReduceOp.SUM)
    avg_accuracy_tensor /= world_size
    
    model.train()
    return avg_accuracy_tensor.item()


# ==========================================================
# 6️⃣ 主入口
# ==========================================================
if __name__ == "__main__":
    """
    主程序入口
    
    关键参数设置:
    - graph_dir: 图数据存储目录
    - num_parts: 图划分份数，对应分布式进程数
    - nodes: 图节点数量
    - world_size: 分布式进程数量
    """
    # 关键参数: 图数据配置
    graph_dir = prepare_graph(graph_dir="datasets/link_prediction_ER", num_parts=3, nodes=1000)
    world_size = 3
    device = None
    
    # 使用mp.spawn启动分布式训练
    mp.spawn(
        train_fn,
        args=(world_size, graph_dir),
        nprocs=world_size,
        join=True
    )
