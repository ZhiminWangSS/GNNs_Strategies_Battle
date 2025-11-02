from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union
import torch
import torch.nn as nn
import time
import os

if TYPE_CHECKING:
    from metrics import MetricsCollector
    from models.gcn import GCN
    from datasets.ogbn_arxiv import OGBNArxiv
    from datasets.ogbl_ddi import OGBLDDI


class BaseStrategy(ABC):
    """Base class for all parallel training strategies"""
    
    def __init__(self, num_workers: int = 1, device: str = 'cuda', **kwargs):
        self.num_workers = num_workers
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.results_dir = f"results/{self.__class__.__name__}"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def _prepare_training(self, model: 'GCN', dataset: Union[OGBNArxiv, OGBLDDI], learning_rate: float):
        """Common training setup"""
        model = model.to(self.device)
        dataset.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        graph = dataset.graph
        labels = dataset.labels
        train_idx = dataset.get_train_indices()
        return model, optimizer, graph, labels, train_idx
    
    def _prepare_evaluation(self, model: 'GCN', dataset: Union[OGBNArxiv, OGBLDDI]):
        """Common evaluation setup"""
        model.eval()
        graph = dataset.graph
        labels = dataset.labels
        test_idx = dataset.get_test_indices()
        return graph, labels, test_idx
    
    def _save_model(self, model: 'GCN', dataset_name: str, epoch: int = None):
        """Save model weights to results folder"""
        if epoch is not None:
            path = os.path.join(self.results_dir, f"{dataset_name}_epoch_{epoch}.pt")
        else:
            path = os.path.join(self.results_dir, f"{dataset_name}_final.pt")
        torch.save(model.state_dict(), path)
        return path
    
    def _load_model(self, model: 'GCN', dataset_name: str, epoch: int = None):
        """Load model weights from results folder"""
        if epoch is not None:
            path = os.path.join(self.results_dir, f"{dataset_name}_epoch_{epoch}.pt")
        else:
            path = os.path.join(self.results_dir, f"{dataset_name}_final.pt")
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=self.device))
            return True
        return False
    
    @abstractmethod
    def train_arxiv(self, model: 'GCN', dataset: 'OGBNArxiv', 
                    epochs: int, learning_rate: float = 0.01, **kwargs) -> 'MetricsCollector':
        """Strategy-specific training for OGBN-arxiv (node classification)"""
        pass
    
    @abstractmethod
    def train_ddi(self, model: 'GCN', dataset: 'OGBLDDI', 
                  epochs: int, learning_rate: float = 0.01, **kwargs) -> 'MetricsCollector':
        """Strategy-specific training for OGBL-ddi (link prediction)"""
        pass
    
    @abstractmethod
    def evaluate_arxiv(self, model: 'GCN', dataset: 'OGBNArxiv') -> float:
        """Strategy-specific evaluation for OGBN-arxiv"""
        pass
    
    @abstractmethod
    def evaluate_ddi(self, model: 'GCN', dataset: 'OGBLDDI') -> float:
        """Strategy-specific evaluation for OGBL-ddi"""
        pass
