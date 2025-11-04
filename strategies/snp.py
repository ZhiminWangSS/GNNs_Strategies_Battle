from strategies.base import BaseStrategy
from models.gcn import GCN
import torch
import dgl
from metrics import MetricsCollector
from datasets.ogbn_arxiv import OGBNArxiv
from datasets.ogbl_ddi import OGBLDDI


class SNP(BaseStrategy):
    """Subgraph Node Parallelism: Partition graph into subgraphs"""
    
    def train_arxiv(self, model: GCN, dataset: OGBNArxiv, 
                    epochs: int, learning_rate: float = 0.01, **kwargs) -> MetricsCollector:
        model, optimizer, graph, labels, train_idx = self._prepare_training(model, dataset, learning_rate)
        
        
        metrics = MetricsCollector()
        model.train()
        
        for epoch in range(epochs):
            pass
        
        self._save_model(model, "arxiv")
        return metrics
    
    def train_ddi(self, model: GCN, dataset: OGBLDDI, 
                  epochs: int, learning_rate: float = 0.01, **kwargs) -> MetricsCollector:
        model, optimizer, graph, labels, train_idx = self._prepare_training(model, dataset, learning_rate)
        train_mask = torch.zeros(graph.num_nodes(), dtype=torch.bool, device=self.device)
        
        metrics = MetricsCollector()
        model.train()
        
        for epoch in range(epochs):
            pass
        
        self._save_model(model, "ddi")
        return metrics
    
    def evaluate_arxiv(self, model: GCN, dataset: OGBNArxiv) -> float:
        graph, labels, test_idx = self._prepare_evaluation(model, dataset)
        
        with torch.no_grad():
            pass
    
    def evaluate_ddi(self, model: GCN, dataset: OGBLDDI) -> float:
        graph, labels, test_idx = self._prepare_evaluation(model, dataset)
        
        with torch.no_grad():
            pass
            