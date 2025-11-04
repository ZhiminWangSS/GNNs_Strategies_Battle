```python
#!/usr/bin/env python3
# ===============================================================
# 生成可控结构特征的图数据 + DGL格式转换 + 单机多卡训练
# ===============================================================
# 主流程：
# 1. 使用 NetworkX 生成图（SBM 或 powerlaw_cluster_graph）
# 2. 为每个节点生成特征（结构特征 + 随机嵌入）
# 3. 生成标签（可基于社区或随机）
# 4. 转为 DGLGraph，并添加 mask（train/val/test）
# 5. 保存为 DGL 图文件
# 6. 使用 DGL 的分布式邻居采样 NodeDataLoader 进行并行训练
# ===============================================================

import argparse
import os
import random
import time
from functools import partial

import networkx as nx
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

import dgl
from dgl.nn import SAGEConv
from dgl.dataloading import MultiLayerNeighborSampler, NodeDataLoader

# ===============================================================
# 一、图生成函数
# ===============================================================

def gen_sbm(n, n_communities, p_in, p_out, seed=None):
    """
    生成随机块模型（Stochastic Block Model, SBM）
    ----------------------------------------------------
    控制参数：
      - n: 节点数
      - n_communities: 社区数（标签类别数）
      - p_in: 社区内连边概率（控制模块度）
      - p_out: 社区间连边概率（越小社区越明显）
    返回：带社区标签的 networkx.Graph
    """
    if seed is not None:
        np.random.seed(seed)

    # 每个社区节点数（均分）
    sizes = [n // n_communities] * n_communities
    sizes[-1] += n - sum(sizes)  # 修正最后一个社区的节点数
    # 构造 block 概率矩阵
    p = [[p_in if i == j else p_out for j in range(n_communities)] for i in range(n_communities)]
    # 生成图
    G = nx.stochastic_block_model(sizes, p, seed=seed)

    # 为每个节点添加社区标签属性 label
    labels = []
    for i, size in enumerate(sizes):
        labels += [i] * size
    nx.set_node_attributes(G, {i: {'label': labels[i]} for i in range(len(labels))})
    return G


def gen_powerlaw_cluster(n, m, p_tri, n_classes=3, seed=None):
    """
    生成 powerlaw + triadic closure 图（Holme-Kim 模型）
    ----------------------------------------------------
    控制参数：
      - n: 节点数
      - m: 每次新加入节点连接的边数（控制平均度）
      - p_tri: 三角闭合概率（控制聚类系数）
      - n_classes: 类别数（标签数）
    返回：带伪社区标签的 networkx.Graph
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 生成图：度分布近似幂律，且可调节聚类
    G = nx.powerlaw_cluster_graph(n, m, p_tri, seed=seed)

    # 社区检测（使用贪心模块度算法）
    from networkx.algorithms import community
    communities = community.greedy_modularity_communities(G)

    # 将社区分配为 n_classes 类别（简单映射）
    labels = np.zeros(n, dtype=np.int64)
    comms = sorted(communities, key=lambda c: -len(c))
    for idx, comm in enumerate(comms):
        cls = idx % n_classes
        for v in comm:
            labels[v] = cls
    nx.set_node_attributes(G, {i: {'label': int(labels[i])} for i in range(n)})
    return G


# ===============================================================
# 二、NetworkX → DGL 转换 + 特征与mask生成
# ===============================================================
def nx_to_dgl_with_feats(G, feat_dim=64, add_struct_feats=True,
                         train_ratio=0.6, val_ratio=0.2, seed=None):
    """
    将 networkx.Graph 转为 DGLGraph，并添加节点特征和mask
    ----------------------------------------------------------
    - feat_dim: 节点总特征维度
    - add_struct_feats: 是否添加结构特征
        包含：
          * degree（度）
          * clustering coefficient（局部聚类系数）
          * pagerank（重要性）
      剩余维度随机补足为随机嵌入
    - mask: train/val/test 按比例划分
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # 确保节点ID为0..n-1（防止SBM返回dict型节点）
    G = nx.convert_node_labels_to_integers(G, ordering='sorted')
    dgl_g = dgl.from_networkx(G)  # 转换为DGLGraph

    n = dgl_g.num_nodes()

    # 取出label属性（节点类别）
    if 'label' in G.nodes[0]:
        labels = torch.LongTensor([G.nodes[i]['label'] for i in range(n)])
    else:
        labels = torch.randint(0, 3, (n,))

    # ---------- 节点特征构造 ----------
    feat_list = []
    if add_struct_feats:
        # 度特征（表示节点连边数量）
        deg = torch.tensor([d for _, d in G.degree()], dtype=torch.float).unsqueeze(1)
        # 聚类系数（反映局部三角形闭合概率）
        clustering = torch.tensor(list(nx.clustering(G).values()), dtype=torch.float).unsqueeze(1)
        # PageRank（衡量节点在图中的全局重要性）
        pr = torch.tensor(list(nx.pagerank(G).values()), dtype=torch.float).unsqueeze(1)
        feat_list += [deg, clustering, pr]

    # 若结构特征维度不够，总维度补齐为 feat_dim
    cur_dim = sum([f.shape[1] for f in feat_list]) if feat_list else 0
    if cur_dim < feat_dim:
        rand_feat = torch.randn(n, feat_dim - cur_dim)
        feat_list.append(rand_feat)
    feat = torch.cat(feat_list, dim=1)

    # ---------- 数据集划分 ----------
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[idx[:n_train]] = True
    val_mask[idx[n_train:n_train + n_val]] = True
    test_mask[idx[n_train + n_val:]] = True

    # ---------- 写入DGL属性 ----------
    dgl_g.ndata['feat'] = feat
    dgl_g.ndata['label'] = labels
    dgl_g.ndata['train_mask'] = train_mask
    dgl_g.ndata['val_mask'] = val_mask
    dgl_g.ndata['test_mask'] = test_mask

    return dgl_g


# ===============================================================
# 三、定义模型（GraphSAGE）
# ===============================================================
class GraphSAGE(nn.Module):
    """基础的GraphSAGE模型"""
    def __init__(self, in_feats, hid_feats, n_classes, num_layers=2, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        # 输入层
        self.layers.append(SAGEConv(in_feats, hid_feats, aggregator_type='mean'))
        # 中间层（可多层）
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hid_feats, hid_feats, aggregator_type='mean'))
        # 输出层
        self.layers.append(SAGEConv(hid_feats, n_classes, aggregator_type='mean'))

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

    def forward(self, blocks, x):
        """
        前向传播（mini-batch模式）
        输入：
          - blocks: DGL采样出的多层邻居子图
          - x: 源节点特征（srcdata['feat']）
        输出：
          - 每个batch目标节点的类别logits
        """
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != self.num_layers - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


# ===============================================================
# 四、分布式训练进程（每GPU一个）
# ===============================================================
def run_worker(rank, world_size, args):
    """
    单个GPU的训练逻辑。
    ---------------------------------------------------
    流程：
      1. 设置CUDA设备 + 初始化通信组（NCCL）
      2. 加载或生成DGL图
      3. 构建模型 + 分布式封装(DDP)
      4. 构建邻居采样DataLoader（每个进程不同节点子集）
      5. 训练并周期性验证
    """
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    # 初始化NCCL通信
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:12355',
        rank=rank,
        world_size=world_size
    )

    # ========== 载入或生成图 ==========
    if rank == 0:
        print(f"[Master] loading graph from {args.graph_path} (or generating)...")

    if args.graph_path and os.path.exists(args.graph_path):
        graphs, _ = dgl.load_graphs(args.graph_path)
        g = graphs[0]
    else:
        # 各进程可独立生成相同随机图（保证seed一致）
        if args.generator == 'sbm':
            nx_g = gen_sbm(args.num_nodes, args.num_classes, args.p_in, args.p_out, seed=args.seed)
        else:
            nx_g = gen_powerlaw_cluster(args.num_nodes, args.m, args.p_tri,
                                        n_classes=args.num_classes, seed=args.seed)
        g = nx_to_dgl_with_feats(nx_g, feat_dim=args.feat_dim,
                                 add_struct_feats=args.add_struct_feats, seed=args.seed)

    # 增加自环
    g = dgl.add_self_loop(g)

    in_feats = g.ndata['feat'].shape[1]
    n_classes = args.num_classes

    # ========== 模型与优化器 ==========
    model = GraphSAGE(in_feats, args.hid_dim, n_classes,
                      num_layers=args.num_layers, dropout=args.dropout).to(device)
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[rank],
                                                output_device=rank)

    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_f = nn.CrossEntropyLoss()

    # ========== DGL采样加载器 ==========
    sampler = MultiLayerNeighborSampler([args.fanout] * (args.num_layers - 1))

    # 取出训练节点索引，并按GPU划分
    train_nids = torch.nonzero(g.ndata['train_mask'], as_tuple=False).squeeze()
    train_nids = train_nids[torch.randperm(train_nids.shape[0],
                                           generator=torch.Generator().manual_seed(args.seed))]
    # 每个rank取间隔采样后的子集
    per_rank_nids = train_nids[rank::world_size]

    dataloader = NodeDataLoader(
        g,
        per_rank_nids,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        device=device
    )

    # ========== 训练循环 ==========
    for epoch in range(1, args.epochs + 1):
        model.train()
        tic = time.time()
        epoch_loss = 0.0

        for input_nodes, output_nodes, blocks in dataloader:
            x = blocks[0].srcdata['feat']
            y = blocks[-1].dstdata['label']
            logits = model(blocks, x)
            loss = loss_f(logits, y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        toc = time.time()
        if rank == 0 and epoch % args.log_every == 0:
            print(f"[Rank {rank}] Epoch {epoch} loss {epoch_loss:.4f} time {(toc-tic):.2f}s")

        # ========== 验证（仅主进程） ==========
        if rank == 0 and epoch % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                feats = g.ndata['feat'].to(device)
                h = feats
                for l, layer in enumerate(model.module.layers):
                    h = layer(g.to(device), h)
                    if l != args.num_layers - 1:
                        h = torch.relu(h)
                logits = h
                labels = g.ndata['label'].to(device)
                val_mask = g.ndata['val_mask'].to(device)
                test_mask = g.ndata['test_mask'].to(device)
                val_acc = (logits[val_mask].argmax(1) == labels[val_mask]).float().mean().item()
                test_acc = (logits[test_mask].argmax(1) == labels[test_mask]).float().mean().item()
                print(f"[Eval] Epoch {epoch} Val Acc: {val_acc:.4f} Test Acc: {test_acc:.4f}")

    dist.barrier()
    dist.destroy_process_group()


# ===============================================================
# 五、主函数入口
# ===============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-nodes', type=int, default=20000)
    parser.add_argument('--num-classes', type=int, default=3)
    parser.add_argument('--generator', choices=['sbm', 'powerlaw'], default='sbm')
    parser.add_argument('--p_in', type=float, default=0.02)
    parser.add_argument('--p_out', type=float, default=0.001)
    parser.add_argument('--m', type=int, default=3)
    parser.add_argument('--p_tri', type=float, default=0.1)
    parser.add_argument('--feat-dim', type=int, default=64)
    parser.add_argument('--add-struct-feats', action='store_true')
    parser.add_argument('--graph-path', type=str, default='generated_graph.bin')
    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--hid-dim', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--fanout', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-every', type=int, default=1)
    parser.add_argument('--eval-every', type=int, default=2)
    args = parser.parse_args()

    # 若不存在图文件则先生成
    if not os.path.exists(args.graph_path):
        print("Generating graph...")
        if args.generator == 'sbm':
            nx_g = gen_sbm(args.num_nodes, args.num_classes, args.p_in, args.p_out, seed=args.seed)
        else:
            nx_g = gen_powerlaw_cluster(args.num_nodes, args.m, args.p_tri,
                                        n_classes=args.num_classes, seed=args.seed)
        g = nx_to_dgl_with_feats(nx_g, feat_dim=args.feat_dim,
                                 add_struct_feats=args.add_struct_feats, seed=args.seed)
        print("Saving graph to", args.graph_path)
        dgl.save_graphs(args.graph_path, [g])
    else:
        print("Graph exists at", args.graph_path)

    # 启动多GPU并行（spawn方式）
    world_size = args.num_gpus
    if world_size > 1:
        mp.spawn(run_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run_worker(0, 1, args)


if __name__ == '__main__':
    main()

```

