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
import datetime

# ==========================================================
# 添加路径
# ==========================================================
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.gcn import GCN
from datasets.data_generator import GraphGenerator


# ==========================================================
# 1️⃣ 初始化分布式环境
# ==========================================================
def setup_distributed(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")
    print(f"🚀 Rank {rank}: 初始化分布式环境")
    return device


# ==========================================================
# 2️⃣ 通信统计模块（hook）
# ==========================================================
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
        duration = max(1e-6, self.end_time - self.start_time)
        mb = self.total_bytes / (1024 * 1024)
        return mb, mb / duration  # (通信量 MB, 带宽 MB/s)


# ==========================================================
# 3️⃣ 评估函数
# ==========================================================
def evaluate(model, test_loader, device, rank, world_size, subg):
    model.eval()
    total_acc = 0.0
    num_batches = 0

    with torch.no_grad():
        for input_nodes, output_nodes, blocks in test_loader:
            feats = subg.ndata["feat"][input_nodes].to(device)
            labels = subg.ndata["labels"][output_nodes].to(device)
            logits = model(blocks, feats)
            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean().item()
            total_acc += acc
            num_batches += 1

    avg_acc = total_acc / num_batches if num_batches > 0 else 0
    acc_tensor = torch.tensor(avg_acc, device=device)
    dist.all_reduce(acc_tensor, op=dist.ReduceOp.AVG)
    return acc_tensor.item()


# ==========================================================
# 4️⃣ 主训练函数
# ==========================================================
def train(rank, local_rank, world_size, device, graph_dir, num_epochs=20, lr=0.001, partition_method="metis"):
    torch.manual_seed(0)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=f"runs/node_cls_rank{rank}_{timestamp}") if rank == 0 else None

    # 数据加载
    gen = GraphGenerator()
    train_loader, test_loader, subg = gen.get_dataloader_for_node_classification(
        pid=rank,
        partition_method=partition_method,
        batch_size=32,
        train_ratio=0.8,
        device=device,
        sampler_fanouts=[10, 5],
        partition_dir=graph_dir,
    )
    subg = subg.to(device)

    # 模型初始化
    input_dim = 4
    hidden_dim = 64
    num_classes = subg.ndata["labels"].max().item() + 1
    model = GCN(input_dim, hidden_dim, num_classes, dropout=0.0).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    comm_monitor = CommunicationMonitor()

    if rank == 0:
        print(f"🚀 模型初始化完成: {sum(p.numel() for p in model.parameters()):,} 参数")

    # ======================================================
    # 🔍 训练循环
    # ======================================================
    for epoch in range(num_epochs):

            model.train()
            total_loss, total_acc = 0.0, 0.0
            num_batches = 0
            epoch_forward, epoch_backward, epoch_comm, epoch_batch = 0.0, 0.0, 0.0, 0.0
            epoch_comm_mb, epoch_comm_bw = 0.0, 0.0
            comm_monitor.start()

            for input_nodes, output_nodes, blocks in train_loader:
                batch_start = time.time()

                # ====== 1️⃣ 数据准备 ======
                feats = subg.ndata["feat"][input_nodes].to(device)
                labels = subg.ndata["labels"][output_nodes].to(device)

                # ====== 2️⃣ 前向传播 ======
                t0 = time.time()
                logits = model(blocks, feats)
                forward_time = time.time() - t0

                # ====== 3️⃣ 损失与反向传播 ======
                loss = nn.functional.cross_entropy(logits, labels)
                t1 = time.time()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                backward_time = time.time() - t1

                # ====== 4️⃣ 通信统计 ======
                comm_monitor.stop()
                comm_mb, comm_bw = comm_monitor.get_bandwidth()

                # ====== 5️⃣ 精度与耗时统计 ======
                preds = logits.argmax(dim=1)
                acc = (preds == labels).float().mean().item()
                batch_time = time.time() - batch_start

                # ====== GPU内存监控 ======
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    gpu_memory_allocated = torch.cuda.memory_allocated(device) / (1024**2)  # MB
                    gpu_memory_reserved = torch.cuda.memory_reserved(device) / (1024**2)  # MB
                    gpu_utilization = GPUtil.getGPUs()[rank].load * 100  # %
                else:
                    gpu_memory_allocated = 0.0
                    gpu_memory_reserved = 0.0
                    gpu_utilization = 0.0

                total_loss += loss.item()
                total_acc += acc
                num_batches += 1
                epoch_forward += forward_time
                epoch_backward += backward_time
                epoch_comm += comm_monitor.end_time - comm_monitor.start_time
                epoch_batch += batch_time
                epoch_comm_bw += comm_bw
                epoch_comm_mb += comm_mb
                if num_batches % 10 == 0 and rank == 0:
                    print(f"[Rank {rank}] Batch {num_batches}: Loss={loss.item():.4f}, Acc={acc:.4f}, BW={comm_bw:.2f}MB/s, GPU Mem={gpu_memory_allocated:.2f}GB, GPU Util={gpu_utilization:.1f}%")



            # ===== 同步平均 =====
            avg_loss = torch.tensor(total_loss / max(1, num_batches), device=device)
            avg_acc = torch.tensor(total_acc / max(1, num_batches), device=device)
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(avg_acc, op=dist.ReduceOp.AVG)

            # if epoch % 10 == 0:
            # ===== 评估 =====
            test_acc = evaluate(model, test_loader, device, rank, world_size, subg)
            # ===== 写入TensorBoard =====
            if rank == 0 and writer:
                writer.add_scalar("Loss/train", avg_loss.item(), epoch)
                writer.add_scalar("Accuracy/train", avg_acc.item(), epoch)
                writer.add_scalar("Accuracy/test", test_acc, epoch)
                writer.add_scalar("Time/forward", epoch_forward / num_batches, epoch)
                writer.add_scalar("Time/backward", epoch_backward / num_batches, epoch)
                writer.add_scalar("Time/comm", epoch_comm / num_batches, epoch)
                writer.add_scalar("Time/batch", epoch_batch / num_batches, epoch)
                writer.add_scalar("Comm/volume_MB", epoch_comm_mb / num_batches, epoch)
                writer.add_scalar("Comm/bandwidth_MBps", epoch_comm_bw / num_batches, epoch)
                writer.add_scalar("GPU/memory_allocated_GB", gpu_memory_allocated, epoch)
                writer.add_scalar("GPU/memory_reserved_GB", gpu_memory_reserved, epoch)
                writer.add_scalar("GPU/utilization_percent", gpu_utilization, epoch)

            if rank == 0:
                print(f"RANK:{rank} - [Epoch {epoch+1}] Loss={avg_loss.item():.4f}, TrainAcc={avg_acc.item():.4f}, TestAcc={test_acc:.4f}")

    if writer:
        writer.close()
    dist.barrier()
    if rank == 0:
        print("✅ 训练完成，所有指标已写入 TensorBoard。")


# ==========================================================
# 5️⃣ 分布式入口
# ==========================================================
def train_fn(rank, world_size, graph_dir, num_epochs=50, lr=0.001):
    device = setup_distributed(rank, world_size)
    train(rank, rank, world_size, device, graph_dir, num_epochs=num_epochs, lr=lr)
    dist.destroy_process_group()


# ==========================================================
# 6️⃣ 图准备 + 启动
# ==========================================================
def prepare_graph(graph_dir="datasets/node_cls", num_parts=3, nodes=1000):
    if not os.path.exists(graph_dir):
        gen = GraphGenerator()
        G_nx = gen.generate_nx_graph(kind="ER", n_nodes=nodes, p=0.01)
        g = gen.nx_to_dgl(G_nx)
        gen.add_node_labels(g)
        gen.partition_graph(g, num_parts=num_parts, method="metis", output_dir=graph_dir)
    return graph_dir


if __name__ == "__main__":
    world_size = 3
    graph_dir = prepare_graph(graph_dir="datasets/node_cls_ER", num_parts=3, nodes=1000)
    num_epochs = 50
    mp.spawn(train_fn, args=(world_size, graph_dir), nprocs=world_size, join=True)
