from typing import List, Tuple, Dict


class MetricsCollector:
    """Simple metrics collector with all metrics in one class"""
    
    def __init__(self, convergence_threshold: float = 0.001, patience: int = 10):
    
        # Dimension 1: Communication Overhead
        self.epoch_times: List[float] = []  # Average epoch time
        self.gpu_communication_times: List[float] = []  # GPU communication time
        self.communication_ratios: List[float] = []  # Communication Ratio
        
        # Dimension 2: Computational Overhead
        self.max_memory_usage: float = 0.0  # Maximum memory usage
        self.gpu_utilization_ratios: List[float] = []  # GPU utilization ratio
        
        # Dimension 3: Convergence
        self.convergence_threshold = convergence_threshold
        self.patience = patience
        self.loss_curve: List[Tuple[int, float]] = []  # Loss curve: (epoch, loss)
        self.converged_at_epoch: int = -1  # Number of epochs to convergence
        
        # Dimension 4: Performance
        self.accuracies: List[float] = []  # Accuracy
        self.roc_aucs: List[float] = []  # ROC-AUC
    
    def _check_convergence(self) -> None:
        """Check if model has converged based on loss curve"""
        if len(self.loss_curve) < self.patience + 1:
            return
        
        recent_losses = [loss for _, loss in self.loss_curve[-self.patience:]]
        if len(recent_losses) < 2:
            return
        
        max_change = max(recent_losses) - min(recent_losses)
        if max_change < self.convergence_threshold and self.converged_at_epoch == -1:
            self.converged_at_epoch = self.loss_curve[-self.patience][0]
    
    # Communication Overhead getters
    def get_average_epoch_time(self) -> float:
        """Get average epoch time"""
        return sum(self.epoch_times) / len(self.epoch_times) if self.epoch_times else 0.0
    
    def get_average_gpu_communication_time(self) -> float:
        """Get average GPU communication time"""
        return sum(self.gpu_communication_times) / len(self.gpu_communication_times) if self.gpu_communication_times else 0.0
    
    def get_average_communication_ratio(self) -> float:
        """Get average communication ratio"""
        return sum(self.communication_ratios) / len(self.communication_ratios) if self.communication_ratios else 0.0
    
    # Computational Overhead getters
    def get_max_memory_usage(self) -> float:
        """Get maximum memory usage"""
        return self.max_memory_usage
    
    def get_average_gpu_utilization(self) -> float:
        """Get average GPU utilization ratio"""
        return sum(self.gpu_utilization_ratios) / len(self.gpu_utilization_ratios) if self.gpu_utilization_ratios else 0.0
    
    # Convergence getters
    def get_epochs_to_convergence(self) -> int:
        """Get number of epochs to convergence"""
        if self.converged_at_epoch == -1:
            return len(self.loss_curve)
        return self.converged_at_epoch
    
    def get_loss_curve(self) -> List[Tuple[int, float]]:
        """Get loss curve"""
        return self.loss_curve.copy()
    
    def get_final_loss(self) -> float:
        """Get final loss"""
        return self.loss_curve[-1][1] if self.loss_curve else 0.0
    
    # Performance getters
    def get_final_accuracy(self) -> float:
        """Get final accuracy"""
        return self.accuracies[-1] if self.accuracies else 0.0
    
    def get_best_accuracy(self) -> float:
        """Get best accuracy"""
        return max(self.accuracies) if self.accuracies else 0.0
    
    def get_final_roc_auc(self) -> float:
        """Get final ROC-AUC"""
        return self.roc_aucs[-1] if self.roc_aucs else 0.0
    
    def get_best_roc_auc(self) -> float:
        """Get best ROC-AUC"""
        return max(self.roc_aucs) if self.roc_aucs else 0.0
    
    def get_all_metrics(self) -> Dict[str, any]:
        """Get all metrics"""
        return {
            'communication_overhead': {
                'average_epoch_time': self.get_average_epoch_time(),
                'average_gpu_communication_time': self.get_average_gpu_communication_time(),
                'average_communication_ratio': self.get_average_communication_ratio()
            },
            'computational_overhead': {
                'max_memory_usage': self.get_max_memory_usage(),
                'average_gpu_utilization': self.get_average_gpu_utilization()
            },
            'convergence': {
                'epochs_to_convergence': self.get_epochs_to_convergence(),
                'final_loss': self.get_final_loss(),
                'loss_curve': self.get_loss_curve(),
                'converged': self.converged_at_epoch != -1
            },
            'performance': {
                'final_accuracy': self.get_final_accuracy(),
                'best_accuracy': self.get_best_accuracy(),
                'final_roc_auc': self.get_final_roc_auc(),
                'best_roc_auc': self.get_best_roc_auc()
            }
        }
    
    def print_summary(self) -> None:
        """Print summary of all metrics"""
        print("\n" + "=" * 70)
        print("METRICS SUMMARY")
        print("=" * 70)
        print("Dimension 1: Communication Overhead")
        print(f"  Average epoch time: {self.get_average_epoch_time():.4f}s")
        print(f"  GPU communication time: {self.get_average_gpu_communication_time():.4f}s")
        print(f"  Communication ratio: {self.get_average_communication_ratio():.4f}")
        print("\nDimension 2: Computational Overhead")
        print(f"  Maximum memory usage: {self.get_max_memory_usage():.2f} GB")
        print(f"  Average GPU utilization: {self.get_average_gpu_utilization():.4f}")
        print("\nDimension 3: Convergence")
        print(f"  Number of epochs to convergence: {self.get_epochs_to_convergence()}")
        print(f"  Final loss: {self.get_final_loss():.6f}")
        print("\nDimension 4: Performance")
        print(f"  Final accuracy: {self.get_final_accuracy():.4f}")
        print(f"  Best accuracy: {self.get_best_accuracy():.4f}")
        if self.roc_aucs:
            print(f"  Final ROC-AUC: {self.get_final_roc_auc():.4f}")
            print(f"  Best ROC-AUC: {self.get_best_roc_auc():.4f}")
        print("=" * 70 + "\n")

