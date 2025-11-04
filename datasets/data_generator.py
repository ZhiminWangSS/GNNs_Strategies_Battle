import networkx as nx
import numpy as np
import torch
import dgl
from typing import Optional, List, Dict
import random
import os, shutil
from sklearn.preprocessing import MinMaxScaler

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
        - kind (str): 图的类型 ('ER', 'BA', 'SBM')。
        - n_nodes (int): 节点数量。
        - p (float): ER图的边生成概率。
        - m (int): BA图的每个新节点连接的边数。
        - sbm_sizes (Optional[List[int]]): SBM图的社区大小列表。
        - sbm_p_in (float): SBM图社区内边的生成概率。
        - sbm_p_out (float): SBM图社区间边的生成概率。
        - seed (Optional[int]): 随机种子。

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
            p_matrix = [[sbm_p_in if i==j else sbm_p_out for j in range(k)] for i in range(k)]
            G = nx.stochastic_block_model(sbm_sizes, p_matrix, seed=seed)
            G = nx.Graph(G)
            G.remove_edges_from(nx.selfloop_edges(G))
        else:
            raise ValueError(f"Unknown kind: {kind}")
        G = nx.convert_node_labels_to_integers(G, ordering='sorted')
        return G

    def nx_to_dgl(self, G: nx.Graph, edge_feat_dim: int = 0) -> dgl.DGLGraph:
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

        # 归一化（每列特征数值范围差距较大）
        scaler = StandardScaler()
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
        src = src.numpy(); dst = dst.numpy()
        import scipy.sparse as sps
        adj = sps.csr_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
        adj = adj + adj.T
        adj.data = np.minimum(adj.data, 1); adj.eliminate_zeros()
        for _ in range(2):
            for u in range(n):
                neigh = adj[u].indices
                if len(neigh) == 0: continue
                if np.random.rand() < homophily:
                    neigh_labels = labels[neigh]
                    labels[u] = int(np.bincount(neigh_labels).argmax())
                else:
                    labels[u] = np.random.randint(0, num_classes)
        g.ndata[label_key] = torch.from_numpy(labels).long()

    def split_data(self, g: dgl.DGLGraph, task: str = 'node', train_ratio: float = 0.7, val_ratio: float = 0.15) -> Dict[str, np.ndarray]:
        """
        划分训练/验证/测试集。

        参数:
        - g (dgl.DGLGraph): 输入的DGL图。
        - task (str): 任务类型 ('node' 或 'edge')。
        - train_ratio (float): 训练集比例。
        - val_ratio (float): 验证集比例。

        返回:
        - Dict[str, np.ndarray]: 包含训练、验证和测试集索引的字典。
        """
        np.random.seed(self.seed)

        if task == 'node':
            n = g.num_nodes()
            idx = np.random.permutation(n)
            n_train = int(train_ratio * n)
            n_val = int(val_ratio * n)
            train_idx = idx[:n_train]
            val_idx = idx[n_train:n_train + n_val]
            test_idx = idx[n_train + n_val:]
            return {'train': train_idx, 'val': val_idx, 'test': test_idx}

        elif task == 'edge':
            e = g.num_edges()
            idx = np.random.permutation(e)
            n_train = int(train_ratio * e)
            n_val = int(val_ratio * e)
            train_idx = idx[:n_train]
            val_idx = idx[n_train:n_train + n_val]
            test_idx = idx[n_train + n_val:]
            return {'train': train_idx, 'val': val_idx, 'test': test_idx}

        else:
            raise ValueError("task must be 'node' or 'edge'")
    def to_dist_graph(
        self,
        g: dgl.DGLGraph,
        num_parts: int = 3,
        graph_name: str = "demo",
        out_dir: str = "./parted_graph",
        balance_edges: bool = True,
        partition_method: str = "metis",  # 可选 "metis" 或 "random"
    ) -> dgl.distributed.DistGraph:
        """
        将DGL图划分为分布式图。

        参数:
        - g (dgl.DGLGraph): 输入的DGL图。
        - num_parts (int): 划分的子图数量。
        - graph_name (str): 图的名称。
        - out_dir (str): 输出目录。
        - balance_edges (bool): 是否平衡边。
        - partition_method (str): 划分方法 ('metis' 或 'random')。

        返回:
        - dgl.distributed.DistGraph: 分布式图。
        """

        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        # 划分图
        print(f"Partitioning graph into {num_parts} parts using '{partition_method}' method...")

        dgl.distributed.partition_graph(
            graph=g,
            graph_name=graph_name,
            num_parts=num_parts,
            out_path=out_dir,
            balance_edges=balance_edges,
            part_method=partition_method  # 指定划分方式
        )

        # 加载 DistGraph
        part_config = os.path.join(out_dir, f"{graph_name}.json")
        dist_g = dgl.distributed.DistGraph(graph_name, part_config=part_config)
        print(f"DistGraph loaded successfully from {part_config}")
        return dist_g

# ---------------------------
#  分布式节点分类加载器
# ---------------------------
def get_dist_node_dataloader(g: dgl.DGLGraph, split_dict: Dict[str, np.ndarray],
                             batch_size: int = 1024, num_workers: int = 0, shuffle: bool = True,
                             sampler: str = 'neighbor', fanouts: List[int] = [10, 10]) -> tuple:
    """
    获取分布式节点分类任务的数据加载器。

    参数:
    - g (dgl.DGLGraph): 输入的DGL图。
    - split_dict (Dict[str, np.ndarray]): 数据集划分字典。
    - batch_size (int): 批量大小。
    - num_workers (int): 工作线程数。
    - shuffle (bool): 是否打乱数据。
    - sampler (str): 采样器类型。
    - fanouts (List[int]): 每层的采样数量。

    返回:
    - tuple: 包含训练、验证和测试数据加载器的元组。
    """
    from dgl.dataloading import DistNodeDataLoader, MultiLayerNeighborSampler
    if sampler == 'neighbor':
        sampler = MultiLayerNeighborSampler(fanouts)
    else:
        raise ValueError("Unsupported sampler")
    train_loader = DistNodeDataLoader(g, train_idx, sampler, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=num_workers)
    val_loader = DistNodeDataLoader(g, val_idx, sampler, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    test_loader = DistNodeDataLoader(g, test_idx, sampler, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


# ---------------------------
#  分布式链路预测加载器
# ---------------------------
def get_dist_edge_dataloader(g: dgl.DGLGraph, split_dict: Dict[str, np.ndarray],
                             batch_size: int = 1024, num_workers: int = 0,
                             sampler: str = 'neighbor', fanouts: List[int] = [10, 10]) -> tuple:
    """
    获取分布式链路预测任务的数据加载器。

    参数:
    - g (dgl.DGLGraph): 输入的DGL图。
    - split_dict (Dict[str, np.ndarray]): 数据集划分字典。
    - batch_size (int): 批量大小。
    - num_workers (int): 工作线程数。
    - sampler (str): 采样器类型。
    - fanouts (List[int]): 每层的采样数量。

    返回:
    - tuple: 包含训练、验证和测试数据加载器的元组。
    """
    from dgl.dataloading import DistEdgeDataLoader, MultiLayerNeighborSampler
    if sampler == 'neighbor':
        sampler = MultiLayerNeighborSampler(fanouts)
    else:
        raise ValueError("Unsupported sampler")
    train_loader = DistEdgeDataLoader(g, split_dict['train'], sampler, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers)
    val_loader = DistEdgeDataLoader(g, split_dict['val'], sampler, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    test_loader = DistEdgeDataLoader(g, split_dict['test'], sampler, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    gen = GraphGenerator(seed=123)
    G_nx = gen.generate_nx_graph(kind='SBM', sbm_sizes=[500, 500, 500, 500], sbm_p_in=0.08, sbm_p_out=0.005)
    g = gen.nx_to_dgl(G_nx, node_feat_dim=32, edge_feat_dim=8)
    gen.add_node_labels(g, num_classes=8, homophily=0.7)

    print("Graph generated:", g)

    # 划分训练/验证/测试集
    split_dict = gen.split_data(g, task='node')

    # 生成 DistGraph，支持跨 GPU 特征聚合
    dist_g = gen.to_dist_graph(g, num_parts=3, partition_method="metis")

    # 获取 DistDataLoader，可搭配 Pytorch DDP框架实现单机多卡的多进程分布式训练
    train_loader, val_loader, test_loader = get_dist_node_dataloader(dist_g, split_dict)

    print("Batch Data:", next(iter(train_loader)))
