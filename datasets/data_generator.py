import networkx as nx
import numpy as np
import torch
import dgl
from typing import Optional, List, Dict
import random
import os
import shutil
from sklearn.preprocessing import MinMaxScaler
import dgl.distributed as dist_dgl
import multiprocessing
import scipy.sparse as sps
import yaml

# ---------------------------
#   Graph Generator
# ---------------------------


class GraphGenerator:
    def __init__(self, seed: Optional[int] = 42):
        """
        初始化图生成器。

        参数:
        - seed (Optional[int]): 随机种子，默认为42。
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def generate_nx_graph(self, kind: str = 'ER', n_nodes: int = 1000, p: float = 0.01, m: int = 2,
                          sbm_sizes: Optional[List[int]] = None,
                          sbm_p_in: float = 0.1, sbm_p_out: float = 0.01, seed: Optional[int] = None) -> nx.Graph:
        """
        生成NetworkX图。

        参数:
        - kind (str): 图的生成模式 ('ER', 'BA', 'SBM')。
        - seed (Optional[int]): 随机种子，确保图可复现。
        for ER and BA
        - n_nodes (int): 节点数量。
        for ER
            - p (float): ER图的边生成概率。
        for BA
        - m (int): BA图的每个新节点连接的边数。
        for SBM
        - sbm_sizes (Optional[List[int]]): SBM图的社区大小列表。
        - sbm_p_in (float): SBM图社区内边的生成概率。
        - sbm_p_out (float): SBM图社区间边的生成概率


        返回:
        - nx.Graph: 生成的NetworkX图。
        """
        seed = seed if seed is not None else self.seed
        if kind.upper() == 'ER':
            G = nx.fast_gnp_random_graph(n_nodes, p, seed=seed)
        elif kind.upper() == 'BA':
            m_eff = min(m, max(1, n_nodes-1))
            G = nx.barabasi_albert_graph(n_nodes, m_eff, seed=seed)
        elif kind.upper() == 'SBM':
            if sbm_sizes is None:
                raise ValueError("SBM requires sbm_sizes")
            k = len(sbm_sizes)
            p_matrix = [
                [sbm_p_in if i == j else sbm_p_out for j in range(k)] for i in range(k)]
            G = nx.stochastic_block_model(sbm_sizes, p_matrix, seed=seed)
            G = nx.Graph(G)
            G.remove_edges_from(nx.selfloop_edges(G))
        else:
            raise ValueError(f"Unknown kind: {kind}")
        G = nx.convert_node_labels_to_integers(G, ordering='sorted')
        return G

    def nx_to_dgl(self, G: nx.Graph) -> dgl.DGLGraph:
        """
        将NetworkX图转换为DGL图，并基于节点局部结构构造特征：
        - 度 (degree)
        - 平均邻居度 (avg_neighbor_degree)
        - 聚类系数 (clustering coefficient)
        - 三角形数 (triangle count)
        """
        g = dgl.from_networkx(G)
        n = g.num_nodes()

        degrees = np.array([G.degree(i) for i in G.nodes()])
        avg_nbr_deg_dict = nx.average_neighbor_degree(G)
        avg_nbr_deg = np.array([avg_nbr_deg_dict[i] for i in G.nodes()])
        clustering_dict = nx.clustering(G)
        clustering = np.array([clustering_dict[i] for i in G.nodes()])
        triangles_dict = nx.triangles(G)
        triangles = np.array([triangles_dict[i] for i in G.nodes()])

        node_features = np.vstack([
            degrees,
            avg_nbr_deg,
            clustering,
            triangles
        ]).T  # shape = (N, 4)

        # 归一化（每列特征数值范围较大）
        scaler = MinMaxScaler()
        node_features = scaler.fit_transform(node_features)

        g.ndata['feat'] = torch.tensor(node_features, dtype=torch.float32)
        g.ndata['orig_id'] = torch.arange(n)

        return g

    def add_node_labels(self, g: dgl.DGLGraph, num_classes: int = 3, homophily: float = 0.8, label_key: str = 'labels') -> None:
        """
        为DGL图添加节点标签。

        参数:
        - g (dgl.DGLGraph): 输入的DGL图。
        - num_classes (int): 标签类别数。
        - homophily (float): 同质性系数。
        - label_key (str): 标签存储的键名。

        返回:
        - None
        """
        n = g.num_nodes()
        labels = np.random.randint(0, num_classes, size=n)
        src, dst = g.edges()
        src = src.numpy()
        dst = dst.numpy()

        adj = sps.csr_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
        adj = adj + adj.T
        adj.data = np.minimum(adj.data, 1)
        adj.eliminate_zeros()
        for _ in range(2):
            for u in range(n):
                neigh = adj[u].indices
                if len(neigh) == 0:
                    continue
                if np.random.rand() < homophily:
                    neigh_labels = labels[neigh]
                    labels[u] = int(np.bincount(neigh_labels).argmax())
                else:
                    labels[u] = np.random.randint(0, num_classes)
        g.ndata[label_key] = torch.from_numpy(labels).long()

    def partition_graph(self, g: dgl.DGLGraph, num_parts: int = 4, method: str = 'metis',
                        output_dir: str = './partitions'):
        """
        对DGL图进行划分。
        method: 'metis', 'random' and 'direct'
        """
        os.makedirs(output_dir, exist_ok=True)
        # parts {part_id: subgraph}
        parts = {}
        if method == 'metis':
            raw_parts = dgl.metis_partition(g, num_parts)

            for pid, subg in raw_parts.items():
                # 原图节点 ID
                orig_nids = subg.ndata[dgl.NID]

                # 拷贝节点特征
                if 'feat' in g.ndata:
                    subg.ndata['feat'] = g.ndata['feat'][orig_nids]
                if 'labels' in g.ndata:
                    subg.ndata['labels'] = g.ndata['labels'][orig_nids]

                # 保存映射关系
                subg.ndata['orig_id'] = orig_nids
                parts[pid] = subg
        elif method == 'random':
            # -----------------------------
            # 随机划分
            # -----------------------------
            n = g.num_nodes()
            node_parts = np.random.randint(0, num_parts, size=n)

            src, dst = g.edges()
            src, dst = src.numpy(), dst.numpy()

            for pid in range(num_parts):
                # 当前分区的节点集合
                part_nodes = np.where(node_parts == pid)[0]

                # 筛选两端都属于该分区的边
                mask = np.isin(src, part_nodes) & np.isin(dst, part_nodes)
                part_src, part_dst = src[mask], dst[mask]

                # 建立旧 -> 新节点映射
                old2new = {nid: i for i, nid in enumerate(part_nodes)}
                mapped_src = np.array([old2new[s] for s in part_src])
                mapped_dst = np.array([old2new[d] for d in part_dst])

                # 构建子图
                subg = dgl.graph((mapped_src, mapped_dst))
                subg.ndata['feat'] = g.ndata['feat'][part_nodes]
                if 'labels' in g.ndata:
                    subg.ndata['labels'] = g.ndata['labels'][part_nodes]
                subg.ndata['orig_id'] = torch.tensor(part_nodes)

                parts[pid] = subg
        elif method == 'direct':
            n = g.num_nodes()
            nodes_per_part = n // num_parts
            all_nodes = np.arange(n)
            for pid in range(num_parts):
                if pid == num_parts - 1:
                    part_nodes = all_nodes[pid * nodes_per_part:]  # 最后一部分包含剩余节点
                else:
                    part_nodes = all_nodes[pid * nodes_per_part:(pid + 1) * nodes_per_part]

                src, dst = g.edges()
                src, dst = src.numpy(), dst.numpy()
                mask = np.isin(src, part_nodes) & np.isin(dst, part_nodes)
                part_src, part_dst = src[mask], dst[mask]
                old2new = {nid: i for i, nid in enumerate(part_nodes)}
                mapped_src = np.array([old2new[s] for s in part_src])
                mapped_dst = np.array([old2new[d] for d in part_dst])
                subg = dgl.graph((mapped_src, mapped_dst))
                subg.ndata['feat'] = g.ndata['feat'][part_nodes]
                if 'labels' in g.ndata:
                    subg.ndata['labels'] = g.ndata['labels'][part_nodes]
                subg.ndata['orig_id'] = torch.tensor(part_nodes)
                parts[pid] = subg
        else:
            raise ValueError(f"Unknown partition method: {method}")

        for pid, subg in parts.items():
            dgl.save_graphs(os.path.join(
                output_dir, f'graph_part{pid}_{method}.dgl'), [subg])

        print(str(method), parts)

        print(
            f"Graph successfully partitioned into {num_parts} parts ({method}) at {output_dir}")

    def get_dataloader_for_node_classification(self, pid, partition_method, batch_size: int = 32, train_ratio=0.8, num_workers=1, device=torch.device("cuda"), sampler_fanouts=[10, 10, 5], partition_dir: str = './partitions'):
        """
        加载指定pid对应的子图并为子图创建适合于node classification任务的DataLoader
        """
        path = os.path.join(
            partition_dir, f'graph_part{pid}_{partition_method}.dgl')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Subgraph {path} not found.")

        subg, _ = dgl.load_graphs(path)
        subg = subg[0]

        # Node sampler: 随机邻居采样 (层数 = fanout 数量)
        sampler = dgl.dataloading.NeighborSampler(sampler_fanouts)

        # 全部节点索引
        all_nodes = torch.arange(subg.num_nodes())
        num_train = int(len(all_nodes) * train_ratio)

        # 随机划分
        perm = torch.randperm(len(all_nodes))
        train_nodes = all_nodes[perm[:num_train]]
        test_nodes = all_nodes[perm[num_train:]]

        # 构建训练 DataLoader
        train_loader = dgl.dataloading.DataLoader(
            subg,
            train_nodes,
            sampler,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers,
            device=device
        )

        # 构建测试 DataLoader
        test_loader = dgl.dataloading.DataLoader(
            subg,
            test_nodes,
            sampler,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            device=device
        )

        print(
            f"Subgraph {pid} -> {len(train_nodes)} train nodes, {len(test_nodes)} test nodes")

        return train_loader, test_loader, subg

    def get_dataloader_for_link_prediction(self, pid, partition_method, batch_size: int = 32, train_ratio=0.8, num_workers=1, device=torch.device("cuda"), sampler_fanouts=[10, 10, 5], partition_dir: str = './partitions'):
        """
        加载指定pid对应的子图并为子图创建适合于link prediction任务的DataLoader。

        参数:
        - pid (int): 分区ID。
        - partition_method (str): 分区方法 ('metis' 或 'random')。
        - batch_size (int): 批量大小。
        - train_ratio (float): 训练集比例。
        - num_workers (int): DataLoader的工作线程数。
        - device (torch.device): 设备 (如 'cuda')。
        - sampler_fanouts (List[int]): 每层的采样数量。
        - partition_dir (str): 分区文件的目录。

        返回:
        - train_loader: 训练集的DataLoader。
        - test_loader: 测试集的DataLoader。
        - subg: 当前分区的子图。
        """
        path = os.path.join(
            partition_dir, f'graph_part{pid}_{partition_method}.dgl')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Subgraph {path} not found.")

        subg, _ = dgl.load_graphs(path)
        subg = subg[0]

        # Edge sampler: 随机邻居采样 (层数 = fanout 数量)
        neg_sampler = dgl.dataloading.negative_sampler.Uniform(2)
        sampler = dgl.dataloading.as_edge_prediction_sampler(
            dgl.dataloading.NeighborSampler(sampler_fanouts), negative_sampler=neg_sampler)

        # 全部边索引
        all_edges = torch.arange(subg.num_edges())
        num_train = int(len(all_edges) * train_ratio)

        # 随机划分
        perm = torch.randperm(len(all_edges))
        train_edges = all_edges[perm[:num_train]]
        test_edges = all_edges[perm[num_train:]]

        # 构建训练 DataLoader
        train_loader = dgl.dataloading.DataLoader(
            subg,
            train_edges,
            sampler,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers,
            device=device
        )

        # 构建测试 DataLoader
        test_loader = dgl.dataloading.DataLoader(
            subg,
            test_edges,
            sampler,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            device=device
        )

        print(
            f"Subgraph {pid} -> {len(train_edges)} train edges, {len(test_edges)} test edges")

        return train_loader, test_loader, subg


if __name__ == "__main__":
    task = 'node_classification'
    batch_size = 8
    device = torch.device("cpu")
    seed = 123
    n_nodes = 2000
    p = 0.01

    gen = GraphGenerator(seed=123)
    G_nx = gen.generate_nx_graph(kind='ER', n_nodes=2000, p=0.01) # 生成可控图
    g = gen.nx_to_dgl(G_nx)
    if task == 'node_classification':
        gen.add_node_labels(g)

        # 划分为 num_parts 个子图
        gen.partition_graph(g, num_parts=3, method='metis',
                            output_dir=f'./tmp/{task}/graph_nodes{n_nodes}_p{p}_parts')
        gen.partition_graph(g, num_parts=3, method='random',
                            output_dir=f'./tmp/{task}/graph_nodes{n_nodes}_p{p}_parts')
        gen.partition_graph(g, num_parts=3, method='direct',
                            output_dir=f'./tmp/{task}/graph_nodes{n_nodes}_p{p}_parts')

        train_loader, test_loader, _ = gen.get_dataloader_for_node_classification(
            pid=0, partition_method='random', partition_dir=f'./tmp/{task}/graph_nodes{n_nodes}_p{p}_parts', batch_size=batch_size, device=device)

        for input_nodes, output_nodes, blocks in train_loader:
            # -----------------------------------
            # input_nodes: 当前 batch 所需的源节点ID
            # 这些节点会被送入 GNN，用于消息传递计算
            print("Input nodes:", input_nodes)

            # -----------------------------------
            # output_nodes: 当前 batch 的目标节点ID
            # 通常对应 GNN 需要计算表示的节点集合（destination nodes）
            print("Output nodes:", output_nodes)

            # -----------------------------------
            # blocks: message flow graph (MFG) 的列表
            # blocks[i] 对应 GNN 的第 i 层
            # 每个 block 包含 src/dst 节点及它们的特征
            # srcdata['feat']: 源节点特征，用于前向计算
            # dstdata['labels']: 目标节点标签（如果有的话）
            print("Blocks:", 
                blocks[0].srcdata['feat'],  # 第一层 block 的源节点特征
                blocks[0].dstdata['labels']  # 第一层 block 的目标节点标签
                )

            # -----------------------------------
            # break: 只查看第一个 batch，避免打印过多信息
            break

    elif task == 'link_prediction':

        # 划分为 num_parts 个子图
        gen.partition_graph(g, num_parts=3, method='metis',
                            output_dir=f'./tmp/{task}/graph_nodes{n_nodes}_p{p}_parts')
        gen.partition_graph(g, num_parts=3, method='random',
                            output_dir=f'./tmp/{task}/graph_nodes{n_nodes}_p{p}_parts')
        gen.partition_graph(g, num_parts=3, method='direct',
                            output_dir=f'./tmp/{task}/graph_nodes{n_nodes}_p{p}_parts')

        train_loader, test_loader, _ = gen.get_dataloader_for_link_prediction(
            pid=0, partition_method='random', partition_dir=f'./tmp/{task}/graph_nodes{n_nodes}_p{p}_parts', batch_size=batch_size, device=device)

        for input_nodes, pos_pair_graph, neg_pair_graph, blocks in train_loader:
            # -----------------------------
            # input_nodes: 本 batch 所需的源节点，用于计算消息传递
            # shape: [num_src_nodes] 或 dict（异构图）
            print("Input nodes:", input_nodes)

            # -----------------------------
            # pos_pair_graph: 本 batch 的正边 subgraph，用于链路预测
            # 包含 src/dst 节点和正边信息（通常保留了 edge label）
            # 可以通过 pos_pair_graph.edges() 获取边索引
            print("Positive pair graph:")
            print("  src nodes:", pos_pair_graph.srcdata[dgl.NID])
            print("  dst nodes:", pos_pair_graph.dstdata[dgl.NID])
            print("  edges:", pos_pair_graph.edges())
            if 'label' in pos_pair_graph.edata:
                print("  edge labels:", pos_pair_graph.edata['label'])

            # -----------------------------
            # neg_pair_graph: 本 batch 的负边 subgraph，用于链路预测
            # 和 pos_pair_graph 结构相同，但边是负样本
            print("Negative pair graph:")
            print("  src nodes:", neg_pair_graph.srcdata[dgl.NID])
            print("  dst nodes:", neg_pair_graph.dstdata[dgl.NID])
            print("  edges:", neg_pair_graph.edges())
            if 'label' in neg_pair_graph.edata:
                print("  edge labels:", neg_pair_graph.edata['label'])

            # -----------------------------
            # blocks: 用于节点表示计算的 message flow graph (MFG)
            # blocks 是一个 list，每层的 block 对应 GNN 的一层
            # src/dst 是本层计算所需的节点
            print("Blocks:")
            for i, block in enumerate(blocks):
                print(f"  Block {i}:")
                print("    src nodes:", block.srcdata[dgl.NID])
                print("    dst nodes:", block.dstdata[dgl.NID])
                print("    src features:", block.srcdata['feat'].shape)
                print("    dst labels:", block.dstdata.get(
                    'labels', 'No labels in block')) # 链路预测没有节点标签
                break  # 只看第一个 block

            break  # 只查看第一个 batch
