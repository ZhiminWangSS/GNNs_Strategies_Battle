import os
os.environ['DGL_DISABLE_GRAPHBYOL'] = '1'

import torch
import torch.optim as optim
from datasets.ogbn_arxiv import OGBNArxiv
from models.gcn import GCN


# Dummy exmapel form cursor, actual implementation needs to use dgl.distributed ?
def train_gdp(num_workers=2, use_distributed=False):

    if use_distributed:
        # DGL distributed setup would go here
        # This requires:
        # 1. dgl.distributed.initialize() - Initialize distributed context
        # 2. dgl.distributed.partition_graph() - Partition graph across nodes
        # 3. Each process loads only its partition from shared storage
        # 4. Use distributed samplers and gather/scatter operations
        raise NotImplementedError(
        )
    
    # Single-machine simulation (current approach)
    dataset = OGBNArxiv()
    graph = dataset.graph
    labels = dataset.labels
    
    print(f"Graph: {graph.num_nodes()} nodes, {graph.num_edges()} edges")
    print(f"Features: {graph.ndata['feat'].shape}")
    print(f"Classes: {dataset.num_classes}\n")
    
    train_nodes = dataset.get_train_indices()
    num_train = len(train_nodes)
    nodes_per_worker = num_train // num_workers
    
    worker_nodes = []
    for i in range(num_workers):
        start = i * nodes_per_worker
        end = start + nodes_per_worker if i < num_workers - 1 else num_train
        worker_nodes.append(train_nodes[start:end])
    
    print(f"Worker node splits:")
    for i, nodes in enumerate(worker_nodes):
        print(f"  Worker {i}: {len(nodes)} nodes")
    
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    hidden_dim = 128
    
    model = GCN(input_dim=num_features, hidden_dim=hidden_dim, output_dim=num_classes, dropout=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    epochs = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset.to(device)
    graph = dataset.graph
    features = graph.ndata['feat']
    labels = dataset.labels
    model = model.to(device)
    
    print(f"\nTraining on {device}...\n")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        optimizer.zero_grad()
        for worker_id in range(num_workers):
            worker_train_nodes = worker_nodes[worker_id].to(device)
            
            logits = model(graph, features)
            
            loss = model.compute_loss(logits, labels, worker_train_nodes)
            
            loss.backward()
            total_loss += loss.item()
        
        optimizer.step()
        
        avg_loss = total_loss / num_workers
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    
    print("\nTraining completed!\n")
    return model, dataset


def test_gdp(model, dataset):
    
    model.eval()
    graph = dataset.graph
    features = graph.ndata['feat']
    labels = dataset.labels
    device = graph.device
    
    with torch.no_grad():
        logits = model(graph, features)
        
        test_nodes = dataset.get_test_indices().to(device)
        accuracy = model.compute_accuracy(logits, labels, test_nodes)
        
        print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Test nodes: {len(test_nodes)}\n")


if __name__ == "__main__":
    # Train
    model, dataset = train_gdp(num_workers=4)
    
    # Test
    test_gdp(model, dataset)
