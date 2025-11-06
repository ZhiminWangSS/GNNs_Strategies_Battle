import os
import sys
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import datetime
from torch.nn.parallel import DistributedDataParallel as DDP
import dgl
from torch.utils.tensorboard import SummaryWriter



sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gcn import GCN
from datasets.data_generator import GraphGenerator



class CommunicationMonitor:
    def __init__(self):
        self.total_bytes = 0
        self.start_time = None
        self.end_time = None

    def start(self):
        self.total_bytes = 0
        self.start_time = time.time()

    def hook(self, tensor):
        """每次通信自动统计数据量"""
        self.total_bytes += tensor.numel() * tensor.element_size()
        return tensor

    def stop(self):
        self.end_time = time.time()

    def get_bandwidth(self):
        if self.start_time is None or self.end_time is None:
            return 0.0, 0.0
        duration = max(1e-6, self.end_time - self.start_time)
        kb = self.total_bytes / (1024)
        return kb, kb / duration  # (通信量 KB, 带宽 KB/s)


# ==========================================================
# 1️⃣ 初始化分布式环境
# ==========================================================
def setup_distributed(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")
    print(f"🚀 Rank {rank}: 初始化分布式环境")
    return device


# ==========================================================
# 🔍 通信统计模块
# ==========================================================









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


def train_fn(rank, world_size, graph_dir, num_epochs=20, lr=0.001):
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
    # print(f"Batch size: {train_loader.batch_size}")
    num_batches = len(train_loader)
    # print(f"每个 epoch 需要迭代 {num_batches} 次")
    # if rank == 0:
    #     print(f"📊 Rank {rank} 加载完成 dataloader（子图 {rank}）")
    subg = subg.to(device)
    # 模型初始化
    input_dim = 4   # 关键参数: 输入特征维度
    hidden_dim = 64  # 关键参数: 隐藏层维度
    output_dim = 32   # 关键参数: 输出维度(链路预测得分)
    
    model = GCN(input_dim, hidden_dim, output_dim, dropout=0.0).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    comm_monitor = CommunicationMonitor()
    def communication_hook(state, bucket: dist.GradBucket):
        tensor = bucket.buffer()
        comm_monitor.hook(tensor)
        fut = dist.all_reduce(tensor, async_op=True).get_future()
        # print(f"[Rank {rank}] Hook triggered with tensor size {tensor.numel()} ({tensor.numel() * tensor.element_size()/1024:.2f} KB)")
        return fut.then(lambda fut: fut.value()[0])
    # 注册通信hook以监控通信量
    model.register_comm_hook(state=None, hook=communication_hook)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if rank == 0:
        print(f"🚀 模型初始化完成: {sum(p.numel() for p in model.parameters()):,} 参数")

    # ======================================================
    # 🔍 训练循环
    # ======================================================
    model.train()
    
    # 同步各进程的批次数量，确保分布式训练同步
    local_num_batches = len(train_loader)
    num_batches_tensor = torch.tensor(local_num_batches, device=device)
    all_num_batches = [torch.zeros_like(num_batches_tensor) for _ in range(world_size)]
    dist.all_gather(all_num_batches, num_batches_tensor)
    max_num_batches = max([x.item() for x in all_num_batches])
    
    for epoch in range(num_epochs):
        model.train()
        total_loss, total_acc = 0.0, 0.0
        num_batches = 0
        epoch_forward, epoch_backward, epoch_comm, epoch_batch = 0.0, 0.0, 0.0, 0.0
        epoch_comm_kb, epoch_comm_bw = 0.0, 0.0
        iter = 1
        if iter <= local_num_batches:
            for input_nodes, pos_pair_graph, neg_pair_graph, blocks in train_loader:
                batch_start = time.time()
                # ============ 数据准备 ============
                feats = subg.ndata['feat'][input_nodes].to(device)
                
                # ============ 前向传播 ============
                forward_start = time.time()
                node_emb = model(blocks, feats)   # 输出目标节点的embedding
                
                
                # 正样本embedding提取和得分计算
                pos_src, pos_dst = pos_pair_graph.edges()
                pos_src_emb = node_emb[pos_src]
                pos_dst_emb = node_emb[pos_dst]
                pos_score = (pos_src_emb * pos_dst_emb).sum(dim=1)
                
                
                # 负样本embedding提取和得分计算
                neg_src, neg_dst = neg_pair_graph.edges()
                neg_src_emb = node_emb[neg_src]
                neg_dst_emb = node_emb[neg_dst]
                neg_score = (neg_src_emb * neg_dst_emb).sum(dim=1)
                
                # 计算链路预测损失
                scores = torch.cat([pos_score, neg_score], dim=0)
                labels = torch.cat([
                    torch.ones_like(pos_score),
                    torch.zeros_like(neg_score)
                ]).to(device)
                loss = nn.functional.binary_cross_entropy_with_logits(scores, labels)
                forward_time = (time.time() - forward_start)
                
                # ============ 反向传播与优化 ============
                backward_start = time.time()
                optimizer.zero_grad()
                comm_monitor.start()
                loss.backward()
                comm_monitor.stop()
                optimizer.step()
                batch_time = time.time() - batch_start
                backward_time = (time.time() - backward_start)

                # 计算准确度
                predictions = (torch.sigmoid(scores) > 0.5).float()
                acc = (predictions == labels).float().mean().item()
                
                
                
                # 获取GPU内存使用情况
                gpu_memory_allocated = 0.0
                gpu_memory_reserved = 0.0
                gpu_utilization = 0.0
                try:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        gpu_memory_allocated = torch.cuda.memory_allocated(device) / (1024**2)  # GB
                        gpu_memory_reserved = torch.cuda.memory_reserved(device) / (1024**2)  # GB
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu_utilization = gpus[local_rank].load * 100  # %
                except:
                    pass
                
                total_loss += loss.item()
                total_acc += acc
                num_batches += 1
                epoch_forward += forward_time
                epoch_backward += backward_time
                epoch_batch += batch_time
                # epoch_comm_mb += comm_mb
                # epoch_comm_bw += comm_bw
                
                if num_batches % 10 == 0:
                    print(f"Rank {rank} batch {num_batches} 内训练中, loss: {loss.item():.4f}, acc: {acc:.4f}")
                iter += 1
        
        # ====== 通信统计 ======
        epoch_comm = comm_monitor.end_time - comm_monitor.start_time
        epoch_comm_kb, epoch_comm_bw = comm_monitor.get_bandwidth()
        print(f"Rank {rank} 通信量: {epoch_comm_kb:.4f} KB, {epoch_comm_bw:.4f} KB/s")
        # 7️⃣ 同步其他进程的批次训练（确保分布式训练同步）
        if iter > local_num_batches:
            for _ in range(max_num_batches - local_num_batches):
                dist.all_reduce(torch.zeros(1, device=device))

        # ======================================================
        # 📊 精度与耗时统计
        # ======================================================
        # 计算平均损失和准确度
        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches   
        # 同步所有进程的损失和准确度
        avg_loss = torch.tensor(avg_loss, device=device)
        avg_acc = torch.tensor(avg_acc, device=device)
        dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(avg_acc, op=dist.ReduceOp.AVG)
        test_acc = evaluate(model, test_loader, device, rank, world_size, subg)
        # ======================================================
        # 📈 TensorBoard 写入
        # ======================================================
        if rank == 0:
            writer.add_scalar("Loss/train", avg_loss.item(), epoch)
            writer.add_scalar("Accuracy/train", avg_acc.item(), epoch)
            writer.add_scalar("Accuracy/test", test_acc, epoch)
            writer.add_scalar("Time/forward", epoch_forward / num_batches, epoch)
            writer.add_scalar("Time/backward", epoch_backward / num_batches, epoch)
            writer.add_scalar("Time/comm", epoch_comm / num_batches, epoch)
            writer.add_scalar("Time/batch", epoch_batch / num_batches, epoch)
            writer.add_scalar("Comm/volume_KB", epoch_comm_kb / num_batches, epoch)
            writer.add_scalar("Comm/bandwidth_KBps", epoch_comm_bw / num_batches, epoch)
            writer.add_scalar("GPU/memory_allocated_MB", gpu_memory_allocated, epoch)
            writer.add_scalar("GPU/memory_reserved_MB", gpu_memory_reserved, epoch)
            writer.add_scalar("GPU/utilization_percent", gpu_utilization, epoch)

        if rank == 0:
                print(f"RANK:{rank} - [Epoch {epoch+1}] Loss={avg_loss.item():.4f}, TrainAcc={avg_acc.item():.4f}, TestAcc={test_acc:.4f}")

    # 关闭 TensorBoard writer
    if rank == 0:
        writer.close()
        print(f"Rank {rank} training completed.")


# ==========================================================
# 🔍 评估函数
# ==========================================================
def evaluate(model, test_loader, device, rank, world_size, subg):
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
            # ============ 数据准备 ============
            feats = subg.ndata['feat'][input_nodes].to(device)
            
            # ============ 前向传播 ============
            node_emb = model(blocks, feats)
            
            # ============ 正样本得分计算 ============
            pos_src, pos_dst = pos_pair_graph.edges()
            pos_src_emb = node_emb[pos_src]
            pos_dst_emb = node_emb[pos_dst]
            pos_score = (pos_src_emb * pos_dst_emb).sum(dim=1)
            
            # ============ 负样本得分计算 ============
            neg_src, neg_dst = neg_pair_graph.edges()
            neg_src_emb = node_emb[neg_src]
            neg_dst_emb = node_emb[neg_dst]
            neg_score = (neg_src_emb * neg_dst_emb).sum(dim=1)
            
            # ============ 损失计算 ============
            scores = torch.cat([pos_score, neg_score], dim=0)
            labels = torch.cat([
                torch.ones_like(pos_score),
                torch.zeros_like(neg_score)
            ]).to(device)
            
            # ============ 准确度计算 ============
            predictions = (torch.sigmoid(scores) > 0.5).float()
            accuracy = (predictions == labels).float().mean().item()
            
            total_accuracy += accuracy
            num_batches += 1
    
    # ============ 分布式同步 ============
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
