from models.gcn import GCN


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

def partition_dgl_graph(g, method, num_parts):
    print(f"⚙️ 执行图划分: {method}")
    if method == "metis":
        partition_graph(g, graph_name, num_parts=num_parts, out_path=graph_dir, part_method="metis")
    elif method == "random":
        partition_graph(g, graph_name, num_parts=num_parts, out_path=graph_dir, part_method="random")
    elif method == "non_uniform":
        node_weights = torch.ones(g.num_nodes())
        node_weights[: int(g.num_nodes() * 0.2)] *= 5  # 前20%节点权重大
        g.ndata["node_weight"] = node_weights
        partition_graph(g, graph_name, num_parts=num_parts, out_path=graph_dir, part_method="metis")
    else:
        raise ValueError("未知的划分方式")
    print(f"✅ 图划分完成: {method}")

# 仅在划分文件不存在时执行划分
if not os.path.exists(os.path.join(graph_dir, graph_name + ".json")):
    partition_dgl_graph(g, partition_method, num_parts)

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

gnn = GNN()
