from strategies.base import BaseStrategy
from models.gcn import GCN
import torch
import dgl
from metrics import MetricsCollector
from datasets.ogbn_arxiv import OGBNArxiv
from datasets.ogbl_ddi import OGBLDDI


class DNP(BaseStrategy):
    """Dynamic Node Parallelism: Dynamic node allocation"""
    
    def train_arxiv(self, model: GCN, dataset: OGBNArxiv, 
                    epochs: int, learning_rate: float = 0.01, **kwargs) -> MetricsCollector:
        model, optimizer, graph, labels, train_idx = self._prepare_training(model, dataset, learning_rate)
        
        metrics = MetricsCollector()
        model.train()
        
        for epoch in range(epochs):
            # TODO: Dynamically allocate nodes (e.g., shuffle), create subgraphs, train
            pass
        
        self._save_model(model, "arxiv")
        return metrics
    
    def train_ddi(self, model: GCN, dataset: OGBLDDI, 
                  epochs: int, learning_rate: float = 0.01, **kwargs) -> MetricsCollector:
        model, optimizer, graph, labels, train_idx = self._prepare_training(model, dataset, learning_rate)
        
        metrics = MetricsCollector()
        model.train()
        
        for epoch in range(epochs):
            # TODO:
            pass
        
        self._save_model(model, "ddi")
        return metrics
    
    def evaluate_arxiv(self, model: GCN, dataset: OGBNArxiv) -> float:
        graph, labels, test_idx = self._prepare_evaluation(model, dataset)
        
        with torch.no_grad():
            # TODO: Dynamically partition test nodes, evaluate, aggregate
            pass
    
    def evaluate_ddi(self, model: GCN, dataset: OGBLDDI) -> float:
        graph, labels, test_idx = self._prepare_evaluation(model, dataset)
        
        with torch.no_grad():
            # TODO: Dynamically partition test edges, evaluate, aggregate
            pass
        