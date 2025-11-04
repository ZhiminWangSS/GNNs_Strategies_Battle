import os
import torch
import torch.nn as nn
import dgl
import dgl.nn.pytorch as dglnn


class GCN(nn.Module):
    """Graph Convolutional Network using DGL's GraphConv
    https://www.dgl.ai/dgl_docs/tutorials/models/1_gnn/1_gcn.html
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.conv1 = dglnn.GraphConv(input_dim, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
    
    def forward(self, g: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
        
        h = self.conv1(g, features)
        h = torch.relu(h)
        
        if self.dropout is not None:
            h = self.dropout(h)
        
        h = self.conv2(g, h)
        
        
        return h
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss on masked nodes"""
        return nn.functional.cross_entropy(logits[mask], labels[mask])
    
    def compute_accuracy(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
        """Compute accuracy on masked nodes"""
        predictions = torch.argmax(logits[mask], dim=1)
        correct = (predictions == labels[mask]).float()
        return correct.mean().item()


if __name__ == "__main__":
    """Example usage of GCN model"""
    import dgl
    
    # Create a simple graph for demonstration
    num_nodes = 10
    src = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    dst = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
    g = dgl.graph((src, dst), num_nodes=num_nodes)
    
    # Create model
    input_dim = 128
    hidden_dim = 64
    output_dim = 40
    model = GCN(input_dim, hidden_dim, output_dim, dropout=0.5)
    
    print(f"GCN Model: {input_dim} -> {hidden_dim} -> {output_dim}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    features = torch.randn(num_nodes, input_dim)
    logits = model(g, features)
    print(f"Output shape: {logits.shape}")
    
    # Compute loss and accuracy (example)
    labels = torch.randint(0, output_dim, (num_nodes,))
    train_mask = torch.ones(num_nodes, dtype=torch.bool)
    
    loss = model.compute_loss(logits, labels, train_mask)
    accuracy = model.compute_accuracy(logits, labels, train_mask)
    
    print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
