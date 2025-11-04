import torch
import dgl
from ogb.linkproppred import DglLinkPropPredDataset


class OGBLDDI:
    """OGBL-ddi dataset for link prediction
    
    Drug-drug interaction network: Undirected graph where edges represent interactions.
    - Each node represents an FDA-approved or experimental drug
    - Task: Predict drug-drug interactions (link prediction)
    """
    
    def __init__(self):
        """Initialize and load OGBL-ddi dataset"""
        dataset = DglLinkPropPredDataset(name='ogbl-ddi')
        self.split_edge = dataset.get_edge_split()
        self.graph = dataset[0]
        
        # Generate node features from degrees (ogbl-ddi has no node features)
        # Store in graph.ndata['feat'] for consistency with OGBN-arxiv
        degrees = self.graph.in_degrees().float().unsqueeze(1)
        feature_dim = 128
        
        # Pad to desired dimension
        if degrees.shape[1] < feature_dim:
            padding = torch.zeros(degrees.shape[0], feature_dim - degrees.shape[1], device=degrees.device)
            features = torch.cat([degrees, padding], dim=1)
        else:
            features = degrees[:, :feature_dim]
        
        self.graph.ndata['feat'] = features
    
    @property
    def num_nodes(self):
        return self.graph.num_nodes() if self.graph else 0
    
    @property
    def num_features(self):
        return self.graph.ndata['feat'].shape[1] if self.graph and 'feat' in self.graph.ndata else 0
    
    @property
    def num_classes(self):
        return 2  # edge exists or not
    
    @property
    def labels(self):
        """Synthetic labels for compatibility (link prediction uses edges)"""
        return torch.zeros(self.graph.num_nodes(), dtype=torch.long)
    
    @property
    def split_idx(self):
        """Adapter: convert split_edge to split_idx format for compatibility"""
        return {
            'train': torch.unique(self.split_edge['train']['edge'][:, 0]),
            'valid': torch.unique(self.split_edge['valid']['edge'][:, 0]),
            'test': torch.unique(self.split_edge['test']['edge'][:, 0])
        }
    
    def get_node_features(self):
        """Get node features from graph (for backward compatibility)"""
        return self.graph.ndata['feat']
    
    def get_train_indices(self):
        """Get training node indices (from train edges)"""
        return torch.unique(self.split_edge['train']['edge'][:, 0])
    
    def get_val_indices(self):
        """Get validation node indices (from val edges)"""
        return torch.unique(self.split_edge['valid']['edge'][:, 0])
    
    def get_test_indices(self):
        """Get test node indices (from test edges)"""
        return torch.unique(self.split_edge['test']['edge'][:, 0])
    
    def get_train_labels(self):
        """Get training edge labels"""
        return torch.ones(self.split_edge['train']['edge'].shape[0])
    
    def get_val_labels(self):
        """Get validation edge labels"""
        return self.split_edge['valid'].get('edge_weight', 
            torch.ones(self.split_edge['valid']['edge'].shape[0]))
    
    def get_test_labels(self):
        """Get test edge labels"""
        return self.split_edge['test'].get('edge_weight',
            torch.ones(self.split_edge['test']['edge'].shape[0]))
    
    def get_train_edges(self):
        """Get training edges [N, 2]"""
        return self.split_edge['train']['edge']
    
    def get_val_edges(self):
        """Get validation edges [N, 2]"""
        return self.split_edge['valid']['edge']
    
    def get_test_edges(self):
        """Get test edges [N, 2]"""
        return self.split_edge['test']['edge']
    
    def to(self, device):
        """Move dataset to device"""
        if self.graph is not None:
            self.graph = self.graph.to(device)
        if self.split_edge is not None:
            for split in ['train', 'valid', 'test']:
                if 'edge' in self.split_edge[split]:
                    self.split_edge[split]['edge'] = self.split_edge[split]['edge'].to(device)
        return self