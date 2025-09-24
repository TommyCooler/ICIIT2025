#!/usr/bin/env python3
"""
Evaluation utilities for anomaly detection
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns


class AnomalyEvaluator:
    """Evaluator for anomaly detection models"""
    
    def __init__(self, model, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize evaluator
        
        Args:
            model: Trained contrastive model
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def compute_reconstruction_error(self, data: torch.Tensor) -> np.ndarray:
        """
        Compute reconstruction error for each sample
        
        Args:
            data: Input data tensor of shape (batch_size, seq_len, features)
            
        Returns:
            Reconstruction errors for each sample
        """
        with torch.no_grad():
            data = data.to(self.device)
            
            # Forward pass to get reconstruction
            outputs = self.model(data, data)  # Use same data for both original and augmented
            reconstructed = outputs['reconstructed']
            
            # Compute MSE reconstruction error for each sample
            mse_errors = torch.mean((data - reconstructed) ** 2, dim=(1, 2))  # (batch_size,)
            
            return mse_errors.cpu().numpy()
    
    def compute_anomaly_scores(self, dataloader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute anomaly scores for entire dataset
        
        Args:
            dataloader: DataLoader containing test data
            
        Returns:
            Tuple of (anomaly_scores, true_labels)
        """
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data, batch_labels in dataloader:
                # Compute reconstruction error
                scores = self.compute_reconstruction_error(batch_data)
                all_scores.extend(scores)
                
                # Extract labels if available
                if batch_labels is not None:
                    if isinstance(batch_labels, torch.Tensor):
                        labels = batch_labels.cpu().numpy()
                    else:
                        labels = batch_labels
                    all_labels.extend(labels)
                else:
                    # If no labels, create dummy labels
                    all_labels.extend([0] * len(scores))
        
        return np.array(all_scores), np.array(all_labels)
    
    def find_optimal_threshold(self, scores: np.ndarray, labels: np.ndarray, 
                             method: str = 'f1') -> float:
        """
        Find optimal threshold for anomaly detection
        
        Args:
            scores: Anomaly scores
            labels: True labels (0: normal, 1: anomaly)
            method: Method to optimize ('f1', 'precision', 'recall', 'auc')
            
        Returns:
            Optimal threshold
        """
        if len(np.unique(labels)) < 2:
            # If no anomalies in labels, use percentile-based threshold
            return np.percentile(scores, 95)  # Top 5% as anomalies
        
        # Sort scores and labels
        sorted_indices = np.argsort(scores)
        sorted_scores = scores[sorted_indices]
        sorted_labels = labels[sorted_indices]
        
        best_threshold = 0
        best_metric = 0
        
        # Try different thresholds
        for i in range(len(sorted_scores)):
            threshold = sorted_scores[i]
            predictions = (scores >= threshold).astype(int)
            
            if method == 'f1':
                if len(np.unique(predictions)) > 1:
                    metric = f1_score(labels, predictions, zero_division=0)
                else:
                    metric = 0
            elif method == 'precision':
                metric = precision_score(labels, predictions, zero_division=0)
            elif method == 'recall':
                metric = recall_score(labels, predictions, zero_division=0)
            elif method == 'auc':
                try:
                    metric = roc_auc_score(labels, scores)
                    break  # AUC doesn't depend on threshold
                except ValueError:
                    metric = 0
            else:
                raise ValueError(f"Unknown method: {method}")
            
            if metric > best_metric:
                best_metric = metric
                best_threshold = threshold
        
        return best_threshold
    
    def evaluate(self, dataloader, threshold: Optional[float] = None, 
                threshold_method: str = 'f1') -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            dataloader: DataLoader containing test data with labels
            threshold: Anomaly detection threshold (if None, will find optimal)
            threshold_method: Method to find optimal threshold
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Compute anomaly scores
        scores, labels = self.compute_anomaly_scores(dataloader)
        
        # Find optimal threshold if not provided
        if threshold is None:
            threshold = self.find_optimal_threshold(scores, labels, threshold_method)
        
        # Make predictions
        predictions = (scores >= threshold).astype(int)
        
        # Compute metrics
        metrics = {}
        
        if len(np.unique(labels)) > 1:  # If we have both normal and anomaly labels
            metrics['precision'] = precision_score(labels, predictions, zero_division=0)
            metrics['recall'] = recall_score(labels, predictions, zero_division=0)
            metrics['f1_score'] = f1_score(labels, predictions, zero_division=0)
            
            try:
                metrics['auc_roc'] = roc_auc_score(labels, scores)
            except ValueError:
                metrics['auc_roc'] = 0.0
            
            try:
                metrics['auc_pr'] = average_precision_score(labels, scores)
            except ValueError:
                metrics['auc_pr'] = 0.0
        else:
            # If no anomalies in labels, use percentile-based evaluation
            metrics['precision'] = 0.0
            metrics['recall'] = 0.0
            metrics['f1_score'] = 0.0
            metrics['auc_roc'] = 0.0
            metrics['auc_pr'] = 0.0
        
        # Additional metrics
        metrics['threshold'] = threshold
        metrics['anomaly_ratio'] = np.mean(predictions)
        metrics['mean_score'] = np.mean(scores)
        metrics['std_score'] = np.std(scores)
        
        return metrics
    
    def plot_anomaly_scores(self, dataloader, save_path: Optional[str] = None):
        """
        Plot anomaly scores distribution
        
        Args:
            dataloader: DataLoader containing test data
            save_path: Path to save plot (if None, will display)
        """
        scores, labels = self.compute_anomaly_scores(dataloader)
        
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Score distribution
        plt.subplot(2, 2, 1)
        plt.hist(scores, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Frequency')
        plt.title('Distribution of Anomaly Scores')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Scores over time
        plt.subplot(2, 2, 2)
        plt.plot(scores, alpha=0.7)
        plt.xlabel('Sample Index')
        plt.ylabel('Reconstruction Error')
        plt.title('Anomaly Scores Over Time')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Scores by label (if available)
        if len(np.unique(labels)) > 1:
            plt.subplot(2, 2, 3)
            normal_scores = scores[labels == 0]
            anomaly_scores = scores[labels == 1]
            
            plt.hist(normal_scores, bins=30, alpha=0.7, label='Normal', color='blue')
            plt.hist(anomaly_scores, bins=30, alpha=0.7, label='Anomaly', color='red')
            plt.xlabel('Reconstruction Error')
            plt.ylabel('Frequency')
            plt.title('Scores by Label')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot 4: ROC curve (if available)
        if len(np.unique(labels)) > 1:
            plt.subplot(2, 2, 4)
            try:
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(labels, scores)
                auc = roc_auc_score(labels, scores)
                
                plt.plot(fpr, tpr, color='darkorange', lw=2, 
                        label=f'ROC curve (AUC = {auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend(loc="lower right")
                plt.grid(True, alpha=0.3)
            except ValueError:
                plt.text(0.5, 0.5, 'ROC curve not available', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('ROC Curve')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def detect_anomalies(self, dataloader, threshold: Optional[float] = None) -> Dict:
        """
        Detect anomalies in dataset
        
        Args:
            dataloader: DataLoader containing data to analyze
            threshold: Anomaly detection threshold
            
        Returns:
            Dictionary containing anomaly detection results
        """
        scores, labels = self.compute_anomaly_scores(dataloader)
        
        if threshold is None:
            threshold = self.find_optimal_threshold(scores, labels)
        
        predictions = (scores >= threshold).astype(int)
        anomaly_indices = np.where(predictions == 1)[0]
        
        return {
            'scores': scores,
            'predictions': predictions,
            'anomaly_indices': anomaly_indices,
            'threshold': threshold,
            'num_anomalies': len(anomaly_indices),
            'anomaly_ratio': len(anomaly_indices) / len(scores)
        }


def create_evaluation_dataloader(dataset_type: str, data_path: str, 
                               window_size: int = 100, batch_size: int = 32,
                               **kwargs):
    """
    Create dataloader for evaluation with labels
    
    Args:
        dataset_type: Type of dataset
        data_path: Path to dataset
        window_size: Size of windows
        batch_size: Batch size
        **kwargs: Additional arguments
        
    Returns:
        DataLoader for evaluation
    """
    from .dataloader import create_dataloaders
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        dataset_type=dataset_type,
        data_path=data_path,
        window_size=window_size,
        stride=window_size,
        batch_size=batch_size,
        **kwargs
    )
    
    # Return test dataloader with labels
    if 'test' in dataloaders:
        return dataloaders['test']
    else:
        # If no test dataloader, use validation
        return dataloaders.get('val', None)


# Example usage
if __name__ == "__main__":
    # This is just for testing the evaluation module
    print("Evaluation module loaded successfully!")
    print("Available classes: AnomalyEvaluator")
    print("Available functions: create_evaluation_dataloader")
