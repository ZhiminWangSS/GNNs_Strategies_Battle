from strategies.base import BaseStrategy
from models.gcn import GCN
from metrics import MetricsCollector
from datasets.ogbn_arxiv import OGBNArxiv
from datasets.ogbl_ddi import OGBLDDI


class GDP(BaseStrategy):
    """Graph Data Parallelism: Split nodes across workers"""
    
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
        
        metrics = MetricsCollector()
        model.train()
        
        for epoch in range(epochs):
            pass
        
        self._save_model(model, "ddi")
        return metrics
    
    def evaluate_arxiv(self, model: GCN, dataset: OGBNArxiv) -> float:
        graph, labels, test_idx = self._prepare_evaluation(model, dataset)
        
    
    def evaluate_ddi(self, model: GCN, dataset: OGBLDDI) -> float:
        graph, labels, test_idx = self._prepare_evaluation(model, dataset)
        
        pass
