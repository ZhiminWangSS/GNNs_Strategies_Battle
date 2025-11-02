import os
import torch
import dgl
from ogb.nodeproppred import DglNodePropPredDataset


class OGBNArxiv:
    """OGBN-arxiv dataset for node classification
    
    Citation network: Directed graph where edges represent citations.
    - 128-dim feature vectors (word embeddings from title/abstract)
    - 40 classes (arXiv subject areas: cs.AI, cs.LG, cs.OS, etc.)
    """
    
    def __init__(self, make_bidirectional: bool = True):
        """Initialize and load OGBN-arxiv dataset
        
        Args:
            make_bidirectional: Convert directed graph to bidirectional for GCN
                               (citation relationships become symmetric for message passing)
        """
        dataset = DglNodePropPredDataset(name='ogbn-arxiv')
        self.split_idx = dataset.get_idx_split()
        
        # Get graph and labels
        g, labels = dataset[0]
        
        # Convert directed to undirected for GCN (standard practice)
        if make_bidirectional:
            g = dgl.to_bidirected(g, copy_ndata=True)
        
        self.graph = g
        self.labels = labels.squeeze()
    
    @property
    def num_nodes(self):
        return self.graph.num_nodes() if self.graph else 0
    
    @property
    def num_features(self):
        return self.graph.ndata['feat'].shape[1] if self.graph else 0
    
    @property
    def num_classes(self):
        return 40  # OGBN-arxiv has 40 classes
    
    def get_train_indices(self):
        """Get training node indices"""
        return self.split_idx['train']
    
    def get_val_indices(self):
        """Get validation node indices"""
        return self.split_idx['valid']
    
    def get_test_indices(self):
        """Get test node indices"""
        return self.split_idx['test']
    
    def get_train_labels(self):
        """Get training labels"""
        return self.labels[self.split_idx['train']]
    
    def get_val_labels(self):
        """Get validation labels"""
        return self.labels[self.split_idx['valid']]
    
    def get_test_labels(self):
        """Get test labels"""
        return self.labels[self.split_idx['test']]
    
    
    def to(self, device):
        """Move dataset to device"""
        if self.graph is not None:
            self.graph = self.graph.to(device)
        if self.labels is not None:
            self.labels = self.labels.to(device)
        return self
