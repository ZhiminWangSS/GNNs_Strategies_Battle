# graph_partition.py
import os
import torch
import dgl
from dgl.distributed import partition_graph

def partition_dgl_graph(g, graph_name, out_path, method='metis', num_parts=4):
    """
    对图进行不同方式划分
    method: 'random' | 'metis' | 'non_uniform'
    """
    os.makedirs(out_path, exist_ok=True)
    print(f"正在执行图划分方式: {method}")

    if method == 'random':
        partition_graph(g, graph_name, num_parts=num_parts,
                        out_path=out_path, part_method='random')

    elif method == 'metis':
        partition_graph(g, graph_name, num_parts=num_parts,
                        out_path=out_path, part_method='metis')

    elif method == 'non_uniform':
        # 非均匀划分：人为调整节点权重
        node_weights = torch.ones(g.num_nodes())
        node_weights[: int(g.num_nodes() * 0.2)] *= 5
        g.ndata['node_weight'] = node_weights
        partition_graph(g, graph_name, num_parts=num_parts,
                        out_path=out_path, part_method='metis')
    else:
        raise ValueError("method 仅支持 'random' | 'metis' | 'non_uniform'")

    print(f"✅ 图划分完成: {method} 模式，共 {num_parts} 个分区。")
