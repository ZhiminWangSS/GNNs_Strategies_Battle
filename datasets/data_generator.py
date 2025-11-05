import networkx as nx
import numpy as np
import torch
import dgl
from typing import Optional, List, Dict
import random
import os, shutil
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
        src = src.numpy(); dst = dst.numpy()
        
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

    # ---------------------------
    # 4. 数据划分
    # ---------------------------
    def split_data(self, g: dgl.DGLGraph, train_ratio: float = 0.7, val_ratio: float = 0.15):
        n = g.num_nodes()
        idx = np.random.permutation(n)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)
        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]
        return {'train': train_idx, 'val': val_idx, 'test': test_idx}

    def partition_graph_for_node_classification(self, g: dgl.DGLGraph, num_parts: int = 4, method: str = 'metis',
                        output_dir: str = './partitions'):
        """
        对DGL图进行划分。
        method: 'metis' 或 'random'
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
            # 自定义随机划分逻辑
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
        else:
            raise ValueError(f"Unknown partition method: {method}")

        for pid, subg in parts.items():
            dgl.save_graphs(os.path.join(output_dir, f'graph_part{pid}_{method}.dgl'), [subg])

        print(str(method), parts)

        print(f"Graph successfully partitioned into {num_parts} parts ({method}) at {output_dir}")
        
    def get_dataloader_for_node_classification(self, pid, partition_method, num_workers=1, device=torch.device("cuda"), sampler_fanouts=[10, 10, 5], partition_dir: str = './partitions', batch_size: int = 32):
        """
        加载指定pid对应的子图并为子图创建 DataLoader
        """
        path = os.path.join(partition_dir, f'graph_part{pid}_{partition_method}.dgl')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Subgraph {path} not found.")

        subg, _ = dgl.load_graphs(path)
        subg = subg[0]

        # Node sampler: 随机邻居采样 (层数 = fanout 数量)
        sampler = dgl.dataloading.NeighborSampler(sampler_fanouts)

        # 节点索引
        node_ids = torch.arange(subg.num_nodes())

        # 创建 DGL 的 DataLoader
        dataloader = dgl.dataloading.DataLoader(
            subg,
            node_ids,
            sampler,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers,
            device=device
        )

        return dataloader, subg


if __name__ == "__main__":
    gen = GraphGenerator()
    G_nx = gen.generate_nx_graph(kind='ER', n_nodes=2000, p=0.01)
    g = gen.nx_to_dgl(G_nx)
    gen.add_node_labels(g)

    # 划分为4个子图（metis）
    gen.partition_graph_for_node_classification(g, num_parts=3, method='metis', output_dir='./tmp/graph_parts')
    gen.partition_graph_for_node_classification(g, num_parts=3, method='random', output_dir='./tmp/graph_parts')

    loader, _ = gen.get_dataloader_for_node_classification(pid=0, partition_method='metis', partition_dir='./tmp/graph_parts')
    
    for input_nodes, output_nodes, blocks in loader:
        print("Input nodes:", input_nodes)
        print("Output nodes:", output_nodes)
        print("Blocks:", blocks)
        break  # 只看第一个 batch
