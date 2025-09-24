#!/usr/bin/env python3
"""
Inference script for contrastive learning model
Tests the model on test data with sliding window and visualizes results
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from typing import Dict, Tuple, List
import pickle

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.contrastive_model import ContrastiveModel
from utils.dataloader import create_dataloaders


class ContrastiveInference:
    """Inference class for contrastive learning model"""
    
    def __init__(self, model_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize inference class
        
        Args:
            model_path: Path to saved model checkpoint
            device: Device to run inference on
        """
        self.device = device
        self.model = None
        self.window_size = None
        self.input_dim = None
        
        # Load model
        self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load model from checkpoint"""
        print(f"Loading model from {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model parameters from checkpoint
        self.input_dim = checkpoint.get('input_dim', 2)
        d_model = checkpoint.get('d_model', 256)
        projection_dim = checkpoint.get('projection_dim', 128)
        nhead = checkpoint.get('nhead', 8)
        transformer_layers = checkpoint.get('transformer_layers', 6)
        tcn_output_dim = checkpoint.get('tcn_output_dim', None)
        tcn_kernel_size = checkpoint.get('tcn_kernel_size', 3)
        tcn_num_layers = checkpoint.get('tcn_num_layers', 3)
        dropout = checkpoint.get('dropout', 0.1)
        temperature = checkpoint.get('temperature', 1.0)
        combination_method = checkpoint.get('combination_method', 'concat')
        
        # Create model
        self.model = ContrastiveModel(
            input_dim=self.input_dim,
            d_model=d_model,
            projection_dim=projection_dim,
            nhead=nhead,
            transformer_layers=transformer_layers,
            tcn_output_dim=tcn_output_dim,
            tcn_kernel_size=tcn_kernel_size,
            tcn_num_layers=tcn_num_layers,
            dropout=dropout,
            temperature=temperature,
            combination_method=combination_method
        )
        
        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully")
        print(f"Input dimension: {self.input_dim}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def sliding_window_inference(self, data: np.ndarray, window_size: int, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform sliding window inference on data
        
        Args:
            data: Input data of shape (features, time_steps)
            window_size: Size of sliding window
            stride: Step size for sliding window
            
        Returns:
            Tuple of (reconstruction_errors, anomaly_scores)
        """
        self.window_size = window_size
        features, time_steps = data.shape
        
        # Calculate number of windows
        n_windows = (time_steps - window_size) // stride + 1
        
        reconstruction_errors = []
        anomaly_scores = []
        
        print(f"Processing {n_windows} windows with stride {stride}")
        
        with torch.no_grad():
            for i in range(n_windows):
                start_idx = i * stride
                end_idx = start_idx + window_size
                
                # Extract window
                window_data = data[:, start_idx:end_idx]  # (features, window_size)
                
                # Convert to tensor and transpose to (window_size, features)
                window_tensor = torch.FloatTensor(window_data.T).unsqueeze(0).to(self.device)  # (1, window_size, features)
                
                # Forward pass through model
                outputs = self.model(window_tensor, window_tensor)  # Use same data for original and augmented
                
                # Get reconstruction
                reconstructed = outputs['reconstructed']  # (1, window_size, features)
                
                # Calculate reconstruction error (MSE per timestep)
                reconstruction_error = torch.mean((window_tensor - reconstructed) ** 2, dim=2)  # (1, window_size)
                reconstruction_error = reconstruction_error.squeeze(0).cpu().numpy()  # (window_size,)
                
                # Calculate anomaly score (mean reconstruction error for the window)
                anomaly_score = np.mean(reconstruction_error)
                
                reconstruction_errors.append(reconstruction_error)
                anomaly_scores.append(anomaly_score)
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{n_windows} windows")
        
        # Convert to numpy arrays
        reconstruction_errors = np.array(reconstruction_errors)  # (n_windows, window_size)
        anomaly_scores = np.array(anomaly_scores)  # (n_windows,)
        
        return reconstruction_errors, anomaly_scores
    
    def create_full_reconstruction(self, data: np.ndarray, reconstruction_errors: np.ndarray, 
                                 window_size: int, stride: int = 1) -> np.ndarray:
        """
        Create full reconstruction by averaging overlapping windows
        
        Args:
            data: Original data
            reconstruction_errors: Reconstruction errors from sliding window
            window_size: Window size used
            stride: Stride used
            
        Returns:
            Full reconstruction of original data
        """
        features, time_steps = data.shape
        n_windows = reconstruction_errors.shape[0]
        
        # Initialize reconstruction arrays
        reconstruction_sum = np.zeros_like(data)
        reconstruction_count = np.zeros(time_steps)
        
        # Accumulate reconstructions
        for i in range(n_windows):
            start_idx = i * stride
            end_idx = start_idx + window_size
            
            # Get original window
            original_window = data[:, start_idx:end_idx]
            
            # Calculate reconstruction (original - error)
            reconstructed_window = original_window - np.sqrt(reconstruction_errors[i])  # Approximate reconstruction
            
            # Add to full reconstruction
            reconstruction_sum[:, start_idx:end_idx] += reconstructed_window
            reconstruction_count[start_idx:end_idx] += 1
        
        # Average overlapping regions
        reconstruction_count = np.maximum(reconstruction_count, 1)  # Avoid division by zero
        full_reconstruction = reconstruction_sum / reconstruction_count
        
        return full_reconstruction
    
    def detect_anomalies(self, anomaly_scores: np.ndarray, threshold_percentile: float = 95) -> np.ndarray:
        """
        Detect anomalies based on anomaly scores
        
        Args:
            anomaly_scores: Anomaly scores for each window
            threshold_percentile: Percentile for threshold (default: 95%)
            
        Returns:
            Boolean array indicating anomalies
        """
        threshold = np.percentile(anomaly_scores, threshold_percentile)
        anomalies = anomaly_scores > threshold
        
        print(f"Anomaly threshold: {threshold:.6f}")
        print(f"Number of anomalous windows: {np.sum(anomalies)}")
        print(f"Anomaly rate: {np.sum(anomalies) / len(anomalies) * 100:.2f}%")
        
        return anomalies
    
    def plot_results(self, data: np.ndarray, reconstruction: np.ndarray, 
                    labels: np.ndarray, anomaly_scores: np.ndarray, 
                    anomalies: np.ndarray, window_size: int, stride: int = 1,
                    save_path: str = None):
        """
        Plot original data, reconstruction, and anomalies
        
        Args:
            data: Original data
            reconstruction: Reconstructed data
            labels: Ground truth labels
            anomaly_scores: Anomaly scores
            anomalies: Detected anomalies
            window_size: Window size
            stride: Stride
            save_path: Path to save plot
        """
        features, time_steps = data.shape
        
        # Create figure with subplots
        fig, axes = plt.subplots(features + 2, 1, figsize=(15, 3 * (features + 2)))
        
        # Plot each feature
        for i in range(features):
            ax = axes[i]
            
            # Plot original data
            time_axis = np.arange(time_steps)
            ax.plot(time_axis, data[i], 'b-', label='Original', alpha=0.7, linewidth=1)
            
            # Plot reconstruction
            ax.plot(time_axis, reconstruction[i], 'g-', label='Reconstruction', alpha=0.7, linewidth=1)
            
            # Highlight anomalies
            if np.any(anomalies):
                # Convert window anomalies to time step anomalies
                anomaly_timesteps = []
                for j, is_anomaly in enumerate(anomalies):
                    if is_anomaly:
                        start_idx = j * stride
                        end_idx = min(start_idx + window_size, time_steps)
                        anomaly_timesteps.extend(range(start_idx, end_idx))
                
                if anomaly_timesteps:
                    ax.scatter(anomaly_timesteps, data[i, anomaly_timesteps], 
                             c='red', s=10, alpha=0.6, label='Detected Anomalies')
            
            # Highlight ground truth anomalies
            if labels is not None and len(labels) == time_steps:
                gt_anomaly_indices = np.where(labels == 1)[0]
                if len(gt_anomaly_indices) > 0:
                    ax.scatter(gt_anomaly_indices, data[i, gt_anomaly_indices], 
                             c='orange', s=15, alpha=0.8, label='Ground Truth Anomalies', marker='x')
            
            ax.set_title(f'Feature {i+1}')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot anomaly scores
        ax = axes[features]
        window_indices = np.arange(len(anomaly_scores))
        ax.plot(window_indices, anomaly_scores, 'purple', label='Anomaly Scores', linewidth=1)
        
        # Highlight detected anomalies
        if np.any(anomalies):
            ax.scatter(window_indices[anomalies], anomaly_scores[anomalies], 
                      c='red', s=20, alpha=0.8, label='Detected Anomalies')
        
        ax.set_title('Anomaly Scores')
        ax.set_ylabel('Score')
        ax.set_xlabel('Window Index')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot ground truth labels
        ax = axes[features + 1]
        if labels is not None and len(labels) == time_steps:
            ax.plot(time_axis, labels, 'orange', label='Ground Truth Labels', linewidth=1)
        else:
            ax.text(0.5, 0.5, 'No ground truth labels available', 
                   transform=ax.transAxes, ha='center', va='center')
        
        ax.set_title('Ground Truth Labels')
        ax.set_ylabel('Label')
        ax.set_xlabel('Time Steps')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def evaluate_performance(self, labels: np.ndarray, anomalies: np.ndarray, 
                           window_size: int, stride: int = 1) -> Dict[str, float]:
        """
        Evaluate anomaly detection performance
        
        Args:
            labels: Ground truth labels
            anomalies: Detected anomalies
            window_size: Window size
            stride: Stride
            
        Returns:
            Dictionary of performance metrics
        """
        if labels is None or len(labels) == 0:
            return {"error": "No ground truth labels available"}
        
        # Convert window-level anomalies to time-step level
        time_steps = len(labels)
        n_windows = len(anomalies)
        
        detected_timesteps = np.zeros(time_steps, dtype=bool)
        for i, is_anomaly in enumerate(anomalies):
            if is_anomaly:
                start_idx = i * stride
                end_idx = min(start_idx + window_size, time_steps)
                detected_timesteps[start_idx:end_idx] = True
        
        # Calculate metrics
        tp = np.sum(detected_timesteps & (labels == 1))
        fp = np.sum(detected_timesteps & (labels == 0))
        fn = np.sum(~detected_timesteps & (labels == 1))
        tn = np.sum(~detected_timesteps & (labels == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn
        }


def main():
    """Main function for inference"""
    parser = argparse.ArgumentParser(description='Contrastive Learning Model Inference')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to test data')
    parser.add_argument('--dataset_type', type=str, default='ecg',
                       choices=['ecg', 'psm', 'nab', 'smap_msl', 'smd'],
                       help='Type of dataset')
    
    # Inference arguments
    parser.add_argument('--window_size', type=int, default=100,
                       help='Window size for sliding window')
    parser.add_argument('--stride', type=int, default=1,
                       help='Stride for sliding window')
    parser.add_argument('--threshold_percentile', type=float, default=95,
                       help='Percentile for anomaly threshold')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='inference_results',
                       help='Directory to save results')
    parser.add_argument('--save_plot', action='store_true',
                       help='Save plot to file')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize inference
    inference = ContrastiveInference(args.model_path)
    
    # Load test data
    print(f"Loading test data from {args.data_path}")
    
    if args.dataset_type == 'ecg':
        # Load ECG data
        test_path = os.path.join(args.data_path, "labeled", "test")
        test_files = [f for f in os.listdir(test_path) if f.endswith('.pkl')]
        
        if not test_files:
            print("No test files found!")
            return
        
        # Load first test file
        test_file = test_files[0]
        test_path_full = os.path.join(test_path, test_file)
        
        with open(test_path_full, 'rb') as f:
            test_data = pickle.load(f)
        
        if isinstance(test_data, list):
            test_data = np.array(test_data)
        
        # Ensure correct shape (features, time_steps)
        if test_data.shape[0] > test_data.shape[1]:
            test_data = test_data.T
        
        # Create dummy labels (ECG is unsupervised)
        labels = None
        
    else:
        # For other datasets, use dataloader
        dataloaders = create_dataloaders(
            dataset_type=args.dataset_type,
            data_path=args.data_path,
            window_size=args.window_size,
            batch_size=1,
            num_workers=0
        )
        
        # Get test data
        test_dataset = dataloaders['test'].dataset
        test_data = test_dataset.data
        labels = test_dataset.labels
    
    print(f"Test data shape: {test_data.shape}")
    
    # Perform inference
    print("Starting inference...")
    reconstruction_errors, anomaly_scores = inference.sliding_window_inference(
        test_data, args.window_size, args.stride
    )
    
    # Create full reconstruction
    reconstruction = inference.create_full_reconstruction(
        test_data, reconstruction_errors, args.window_size, args.stride
    )
    
    # Detect anomalies
    anomalies = inference.detect_anomalies(anomaly_scores, args.threshold_percentile)
    
    # Evaluate performance
    if labels is not None:
        performance = inference.evaluate_performance(
            labels, anomalies, args.window_size, args.stride
        )
        print("\nPerformance Metrics:")
        for metric, value in performance.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
    
    # Plot results
    plot_path = os.path.join(args.output_dir, 'inference_results.png') if args.save_plot else None
    inference.plot_results(
        test_data, reconstruction, labels, anomaly_scores, 
        anomalies, args.window_size, args.stride, plot_path
    )
    
    # Save results
    results = {
        'reconstruction_errors': reconstruction_errors,
        'anomaly_scores': anomaly_scores,
        'anomalies': anomalies,
        'reconstruction': reconstruction,
        'performance': performance if labels is not None else None
    }
    
    results_path = os.path.join(args.output_dir, 'inference_results.npz')
    np.savez(results_path, **results)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
