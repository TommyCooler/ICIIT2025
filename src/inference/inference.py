#!/usr/bin/env python3
"""
Inference script for contrastive learning model
Tests the model on test data with sliding window and visualizes results
"""

import torch
import torch.nn as nn
import numpy as np
# import matplotlib.pyplot as plt  # DISABLED: No plotting for automation
import os
import sys
import argparse
from typing import Dict, Tuple, List
import pickle
import pandas as pd

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.contrastive_model import ContrastiveModel
from utils.dataloader import create_dataloaders


def adjustment(gt, pred):
    """
    Point Adjustment algorithm for anomaly detection evaluation.
    Expands predicted anomalies to cover the entire ground truth anomaly region
    when at least one point in the region is correctly predicted.
    """
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            # Expand backward
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            # Expand forward
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def binary_classification_metrics(y_true, y_pred, zero_division=0.0):
    """
    Calculate accuracy, precision, recall, f1 for binary classification.
    - zero_division: value used when denominator = 0 (e.g., no predicted positives)
    """
    yt = np.asarray(y_true).astype(bool)
    yp = np.asarray(y_pred).astype(bool)

    tp = np.sum( yt &  yp)
    fp = np.sum(~yt &  yp)
    fn = np.sum( yt & ~yp)
    tn = np.sum(~yt & ~yp)

    precision = tp / (tp + fp) if (tp + fp) > 0 else zero_division
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return accuracy, precision, recall, f1, (tp, fp, fn, tn)


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
        
        # Load config from config.json in the same directory
        config_path = os.path.join(os.path.dirname(model_path), 'config.json')
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            # Get input_dim from config
            self.input_dim = config.get('input_dim', 2)
            d_model = config.get('d_model', 256)
            projection_dim = config.get('projection_dim', 128)
            nhead = config.get('nhead', 8)
            transformer_layers = config.get('transformer_layers', 6)
            tcn_output_dim = config.get('tcn_output_dim', None)
            tcn_kernel_size = config.get('tcn_kernel_size', 3)
            tcn_num_layers = config.get('tcn_num_layers', 3)
            dropout = config.get('dropout', 0.1)
            temperature = config.get('temperature', 1.0)
            combination_method = config.get('combination_method', 'concat')
            use_contrastive = config.get('use_contrastive', True)
            # Load window_size from config to ensure consistency with training
            self.window_size = config.get('window_size', 128)
            # Load batch_size from config to ensure consistency with training
            self.batch_size = config.get('batch_size', 32)
            # Augmentation-specific hyperparameters (optional)
            self.aug_kwargs = {}
            if 'aug_nhead' in config and config['aug_nhead'] is not None:
                self.aug_kwargs['nhead'] = config.get('aug_nhead')
            if 'aug_num_layers' in config and config['aug_num_layers'] is not None:
                self.aug_kwargs['num_layers'] = config.get('aug_num_layers')
            if 'aug_tcn_kernel_size' in config and config['aug_tcn_kernel_size'] is not None:
                self.aug_kwargs['tcn_kernel_size'] = config.get('aug_tcn_kernel_size')
            if 'aug_tcn_num_layers' in config and config['aug_tcn_num_layers'] is not None:
                self.aug_kwargs['tcn_num_layers'] = config.get('aug_tcn_num_layers')
            if 'aug_dropout' in config and config['aug_dropout'] is not None:
                # Augmentation dropout overrides model dropout
                dropout = config.get('aug_dropout', dropout)
            if 'aug_temperature' in config and config['aug_temperature'] is not None:
                temperature = config.get('aug_temperature', temperature)
            # Load weights from config
            self.contrastive_weight = config.get('contrastive_weight', 1.0)
            self.reconstruction_weight = config.get('reconstruction_weight', 1.0)
            # Load other training parameters
            self.learning_rate = config.get('learning_rate', 1e-4)
            self.weight_decay = config.get('weight_decay', 1e-5)
            self.epsilon = config.get('epsilon', 1e-5)
            self.mask_mode = config.get('mask_mode', 'time')
            self.mask_ratio = config.get('mask_ratio', 0.2)
            self.mask_seed = config.get('mask_seed', None)
            self.device_name = config.get('device', 'cuda')
            self.seed = config.get('seed', 42)
            print(f"Config loaded from {config_path}")
            print(f"Using input_dim: {self.input_dim}")
            print(f"Using window_size: {self.window_size}")
            print(f"Using batch_size: {self.batch_size}")
            print(f"Using contrastive: {use_contrastive}")
        else:
            # Fallback: try to get from checkpoint
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
            use_contrastive = checkpoint.get('use_contrastive', True)
            # Load window_size from checkpoint if available
            self.window_size = checkpoint.get('window_size', 128)
            # Load batch_size from checkpoint if available
            self.batch_size = checkpoint.get('batch_size', 32)
            # Default: no aug overrides from checkpoint unless config provided
            self.aug_kwargs = {}
            # Load weights from checkpoint if available
            self.contrastive_weight = checkpoint.get('contrastive_weight', 1.0)
            self.reconstruction_weight = checkpoint.get('reconstruction_weight', 1.0)
            # Load other training parameters from checkpoint
            self.learning_rate = checkpoint.get('learning_rate', 1e-4)
            self.weight_decay = checkpoint.get('weight_decay', 1e-5)
            self.epsilon = checkpoint.get('epsilon', 1e-5)
            self.mask_mode = checkpoint.get('mask_mode', 'time')
            self.mask_ratio = checkpoint.get('mask_ratio', 0.2)
            self.mask_seed = checkpoint.get('mask_seed', None)
            self.device_name = checkpoint.get('device', 'cuda')
            self.seed = checkpoint.get('seed', 42)
            print("Config file not found, using checkpoint defaults")
            print(f"Using input_dim: {self.input_dim} (overridden for ECG)")
            print(f"Using window_size: {self.window_size}")
            print(f"Using batch_size: {self.batch_size}")
            print(f"Using contrastive: {use_contrastive}")
        
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
            combination_method=combination_method,
            use_contrastive=use_contrastive,
            augmentation_kwargs=self.aug_kwargs if hasattr(self, 'aug_kwargs') else None
        )
        
        # Load state dict with backward-compat mapping for augmentation TCN
        state_dict = checkpoint['model_state_dict']
        mapped_state_dict = {}
        for k, v in state_dict.items():
            # Map old single-layer conv names to new stacked net.0.* names
            if k.startswith('augmentation.tcn_module.conv.'):
                new_k = k.replace('augmentation.tcn_module.conv.', 'augmentation.tcn_module.net.0.')
                mapped_state_dict[new_k] = v
            else:
                mapped_state_dict[k] = v
        missing, unexpected = self.model.load_state_dict(mapped_state_dict, strict=False)
        if missing:
            print(f"Warning: Missing keys when loading state_dict: {missing}")
        if unexpected:
            print(f"Warning: Unexpected keys when loading state_dict: {unexpected}")
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully")
        print(f"Input dimension: {self.input_dim}")
        print(f"Window size: {self.window_size}")
        print(f"Batch size: {self.batch_size}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def sliding_window_inference(self, data: np.ndarray, window_size: int, stride: int = 1, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform sliding window inference on data with batch processing
        
        Note: This method computes absolute errors for all windows, but the final
        timestep scores are computed using a new strategy in compute_timestep_scores():
        - Window 0: Uses all timesteps in the window
        - Window 1+: Only uses the last timestep (since stride=1)
        - No averaging: Each timestep is computed only once
        
        Args:
            data: Input data of shape (features, time_steps)
            window_size: Size of sliding window
            stride: Step size for sliding window
            batch_size: Batch size for processing windows (similar to training)
            
        Returns:
            Tuple of (reconstruction_errors, anomaly_scores, absolute_errors)
        """
        self.window_size = window_size
        _ , time_steps = data.shape
        
        # Calculate number of windows to cover all timesteps
        # We need enough windows so that the last window covers the last timestep
        # Last window should end at time_steps-1, so start at time_steps-window_size
        # Number of windows = (time_steps - window_size) + 1
        n_windows = time_steps - window_size + 1
        
        reconstruction_errors = []
        anomaly_scores = []
        absolute_errors = []
        
        print(f"Processing {n_windows} windows with stride {stride} and batch size {batch_size}")
        print(f"Data length: {time_steps}, Window size: {window_size}")
        print(f"Last window will cover timesteps [{n_windows-1}*{stride}, {(n_windows-1)*stride + window_size}] = [{((n_windows-1)*stride)}, {((n_windows-1)*stride + window_size-1)}]")
        
        # Process windows in batches
        with torch.no_grad():
            for batch_start in range(0, n_windows, batch_size):
                batch_end = min(batch_start + batch_size, n_windows)
                current_batch_size = batch_end - batch_start
                
                # Prepare batch data
                batch_windows = []
                batch_indices = []
                
                for i in range(batch_start, batch_end):
                    start_idx = i * stride
                    end_idx = start_idx + window_size
                    
                    # Check if this window reaches the end of data
                    if end_idx > time_steps:
                        print(f"Window {i}: end_idx ({end_idx}) > time_steps ({time_steps}), stopping")
                        break
                    
                    # Extract window
                    window_data = data[:, start_idx:end_idx]  # (features, window_size)
                    batch_windows.append(window_data.T)  # (window_size, features)
                    batch_indices.append(i)
                
                if not batch_windows:
                    break
                
                # Convert to tensor batch: (batch_size, window_size, features)
                batch_tensor = torch.FloatTensor(np.array(batch_windows)).to(self.device)
                
                # Forward pass through model (no masking during inference)
                outputs = self.model(batch_tensor, batch_tensor)
                
                # Get reconstruction
                reconstructed = outputs['reconstructed']  # (batch_size, window_size, features)
                
                # Calculate absolute error |Eo - Ep| per timestep
                absolute_error = torch.abs(batch_tensor - reconstructed)  # (batch_size, window_size, features)
                
                # Calculate mean absolute error per timestep (across features)
                mean_absolute_error = torch.mean(absolute_error, dim=2)  # (batch_size, window_size)
                mean_absolute_error = mean_absolute_error.cpu().numpy()  # (batch_size, window_size)
                
                # Calculate anomaly score (mean absolute error for each window)
                anomaly_scores_batch = np.mean(mean_absolute_error, axis=1)  # (batch_size,)
                
                # Also keep the old MSE calculation for backward compatibility
                reconstruction_error = torch.mean((batch_tensor - reconstructed) ** 2, dim=2)  # (batch_size, window_size)
                reconstruction_error = reconstruction_error.cpu().numpy()  # (batch_size, window_size)
                
                # Store results for this batch
                for j, window_idx in enumerate(batch_indices):
                    reconstruction_errors.append(reconstruction_error[j])
                    anomaly_scores.append(anomaly_scores_batch[j])
                    absolute_errors.append(mean_absolute_error[j])
                
                if (batch_end) % (batch_size * 10) == 0 or batch_end == n_windows:
                    print(f"Processed {batch_end}/{n_windows} windows")
        
        # Convert to numpy arrays
        reconstruction_errors = np.array(reconstruction_errors)  # (n_windows, window_size)
        anomaly_scores = np.array(anomaly_scores)  # (n_windows,)
        absolute_errors = np.array(absolute_errors)  # (n_windows, window_size)
        
        return reconstruction_errors, anomaly_scores, absolute_errors
    
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
        _ , time_steps = data.shape
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

    def compute_timestep_scores(self, absolute_errors: np.ndarray, time_steps: int,
                                window_size: int, stride: int = 1) -> np.ndarray:
        """
        Compute per-timestep scores using new strategy:
        - Window 0 [0:window_size-1]: Use all timesteps in the window
        - Window 1+ [i:i+window_size-1]: Only use the last timestep (i+window_size-1)
        - This ensures every timestep gets exactly one score
        
        Args:
            absolute_errors: (n_windows, window_size) - absolute errors for each window
            time_steps: Total number of timesteps in original data
            window_size: Size of each window
            stride: Step size between windows
            
        Returns:
            timestep_scores: (time_steps,) - anomaly score for each timestep
        """
        n_windows = absolute_errors.shape[0]
        timestep_scores = np.zeros(time_steps, dtype=np.float32)
        
        print(f"Computing timestep scores for {time_steps} timesteps from {n_windows} windows...")
        print(f"Strategy: Window 0 uses all timesteps, subsequent windows use only last timestep")
        
        for i in range(n_windows):
            start_idx = i * stride
            end_idx = min(start_idx + window_size, time_steps)
            valid_len = end_idx - start_idx
            if valid_len <= 0:
                continue
            
            if i == 0:
                # First window [0:window_size-1]: use all timesteps
                timestep_scores[start_idx:end_idx] = absolute_errors[i][:valid_len]
                print(f"Window {i}: Used all {valid_len} timesteps from {start_idx} to {end_idx-1}")
            else:
                # Subsequent windows [i:i+window_size-1]: only use the last timestep
                last_timestep_idx = end_idx - 1
                if last_timestep_idx < time_steps:
                    timestep_scores[last_timestep_idx] = absolute_errors[i][valid_len - 1]
                    # print(f"Window {i}: Used only last timestep {last_timestep_idx} (value: {absolute_errors[i][valid_len - 1]:.6f})")
        
        # Verify all timesteps are covered
        uncovered_timesteps = np.sum(timestep_scores == 0)
        if uncovered_timesteps > 0:
            print(f"Warning: {uncovered_timesteps} timesteps were not covered by any window")
            # This should not happen with the correct window calculation
        else:
            print(f"All {time_steps} timesteps are covered by windows")
        
        print(f"Timestep scores computed: {len(timestep_scores)} timesteps")
        print(f"Score range: [{np.min(timestep_scores):.6f}, {np.max(timestep_scores):.6f}]")
        print(f"Non-zero scores: {np.sum(timestep_scores != 0)}")
        
        return timestep_scores
    
    def detect_anomalies(self, anomaly_scores: np.ndarray, threshold: float) -> Tuple[float, np.ndarray]:
        """
        Detect anomalies based on anomaly scores
        
        Args:
            anomaly_scores: Anomaly scores for each window
            threshold: Threshold value for anomaly detection
            
        Returns:
            Tuple of (threshold_value, anomalies_boolean_array)
        """
        anomalies = anomaly_scores > threshold
        
        print(f"Anomaly threshold: {threshold:.6f}")
        print(f"Number of anomalous windows: {np.sum(anomalies)}")
        print(f"Anomaly rate: {np.sum(anomalies) / len(anomalies) * 100:.2f}%")
        
        return threshold, anomalies
    
    def create_threshold_range(self, absolute_errors: np.ndarray, num_thresholds: int = 100) -> np.ndarray:
        """
        Create threshold range from [0, max(|Eo-Ep|)]
        
        Args:
            absolute_errors: Absolute errors from inference (n_windows, window_size)
            num_thresholds: Number of threshold values to generate
            
        Returns:
            Array of threshold values
        """
        max_error = np.max(absolute_errors)
        min_error = 0.0
        
        print(f"Number of thresholds: {num_thresholds}")
        
        # Create evenly spaced thresholds
        thresholds = np.linspace(min_error, max_error, num_thresholds)
        
        return thresholds
    
    def evaluate_threshold_range(self, timestep_scores: np.ndarray, labels: np.ndarray,
                                num_thresholds: int = 1000, use_adjustment: bool = True) -> Dict:
        """
        Evaluate performance across a range of threshold values based on timestep anomaly scores
        
        Args:
            timestep_scores: Timestep-level anomaly scores (shape: time_steps,)
            labels: Ground truth labels (shape: time_steps,)
            num_thresholds: Number of threshold values to test (500-1000 recommended)
            use_adjustment: Whether to apply Point Adjustment
            
        Returns:
            Dictionary containing results for all thresholds
        """
        if labels is None or len(labels) == 0:
            return {"error": "No ground truth labels available"}
        
        # Create threshold range from [0, max(anomaly_score)]
        max_score = np.max(timestep_scores)
        min_score = 0.0  # Start from 0 as requested
        thresholds = np.linspace(min_score, max_score, num_thresholds)
        
        print(f"Evaluating {num_thresholds} thresholds")
        
        results = {
            'thresholds': thresholds,
            'precision': [],
            'recall': [],
            'f1_score': [],
            'accuracy': [],
            'true_positives': [],
            'false_positives': [],
            'false_negatives': [],
            'true_negatives': []
        }
        
        best_f1 = 0
        best_threshold = 0
        best_metrics = {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'accuracy': 0.0,
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'tn': 0
        }
        
        # Convert labels to binary arrays
        gt = labels.astype(int)
        
        for i, threshold in enumerate(thresholds):
            # Detect anomalies at timestep level
            pred_before_pa = (timestep_scores > threshold).astype(int)
            
            # Apply Point Adjustment if requested
            if use_adjustment:
                _, pred_after_pa = adjustment(gt.copy(), pred_before_pa.copy())
            else:
                pred_after_pa = pred_before_pa
            
            # Calculate metrics after Point Adjustment
            acc, precision, recall, f1, (tp, fp, fn, tn) = binary_classification_metrics(gt, pred_after_pa)
            
            # Store results
            results['precision'].append(precision)
            results['recall'].append(recall)
            results['f1_score'].append(f1)
            results['accuracy'].append(acc)
            results['true_positives'].append(tp)
            results['false_positives'].append(fp)
            results['false_negatives'].append(fn)
            results['true_negatives'].append(tn)
            
            # Track best F1 score
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,  # Changed from 'f1_score' to 'f1'
                    'accuracy': acc,
                    'tp': tp,  # Changed from 'true_positives' to 'tp'
                    'fp': fp,  # Changed from 'false_positives' to 'fp'
                    'fn': fn,  # Changed from 'false_negatives' to 'fn'
                    'tn': tn   # Changed from 'true_negatives' to 'tn'
                }
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{num_thresholds} thresholds (F1: {f1:.4f})")
        
        # Add best results
        results['best_threshold'] = best_threshold
        results['best_f1'] = best_f1
        results['best_metrics'] = best_metrics
        
        print(f"\n" + "="*80)
        print(f"THRESHOLD ANALYSIS RESULTS")
        print(f"="*80)
        print(f"Total thresholds evaluated: {num_thresholds}")
        print(f"Best threshold: {best_threshold:.6f}")
        print(f"Best F1-score: {best_f1:.4f}")
        print(f"Best precision: {best_metrics['precision']:.4f}")
        print(f"Best recall: {best_metrics['recall']:.4f}")
        print(f"Best accuracy: {best_metrics['accuracy']:.4f}")
        print(f"Best TP: {best_metrics['tp']}")
        print(f"Best FP: {best_metrics['fp']}")
        print(f"Best FN: {best_metrics['fn']}")
        print(f"Best TN: {best_metrics['tn']}")
        print(f"="*80)
        
        return results
    
    def save_threshold_results_to_excel(self, threshold_results: Dict, save_path: str, config_info: Dict = None):
        """
        Save threshold analysis results to Excel file with multiple sheets
        If file exists, append results to existing file to compare multiple models
        
        Args:
            threshold_results: Results from evaluate_threshold_range
            save_path: Path to save Excel file
            config_info: Configuration information to save
        """
        if "error" in threshold_results:
            print(f"Cannot save to Excel: {threshold_results['error']}")
            return
        
        # Extract model timestamp from model path for sheet naming
        model_timestamp = "unknown"
        if config_info and 'Model_Path' in config_info:
            model_path = config_info['Model_Path']
            # Extract timestamp from path like: checkpoints/ecg_20250929_143022/best_model.pth
            import re
            match = re.search(r'(\d{8}_\d{6})', model_path)
            if match:
                model_timestamp = match.group(1)
        
        # Create sheet name with timestamp
        sheet_name = f"Model_{model_timestamp}"
        
        print(f"Saving threshold analysis results to Excel: {save_path}")
        print(f"Sheet name: {sheet_name}")
        
        # Create DataFrame with all threshold results
        df_results = pd.DataFrame({
            'Threshold': threshold_results['thresholds'],
            'Precision': threshold_results['precision'],
            'Recall': threshold_results['recall'],
            'F1_Score': threshold_results['f1_score'],
            'Accuracy': threshold_results['accuracy'],
            'True_Positives': threshold_results['true_positives'],
            'False_Positives': threshold_results['false_positives'],
            'False_Negatives': threshold_results['false_negatives'],
            'True_Negatives': threshold_results['true_negatives']
        })
        
        # Create summary DataFrame with model info
        best_metrics = threshold_results['best_metrics']
        summary_data = {
            'Metric': ['Model_Timestamp', 'Best_Threshold', 'Best_F1_Score', 'Best_Precision', 'Best_Recall', 'Best_Accuracy',
                      'Best_True_Positives', 'Best_False_Positives', 'Best_False_Negatives', 'Best_True_Negatives',
                      'Total_Thresholds'],
            'Value': [
                model_timestamp,
                threshold_results['best_threshold'],
                best_metrics['f1'],
                best_metrics['precision'],
                best_metrics['recall'],
                best_metrics['accuracy'],
                best_metrics['tp'],
                best_metrics['fp'],
                best_metrics['fn'],
                best_metrics['tn'],
                len(threshold_results['thresholds'])
            ]
        }
        df_summary = pd.DataFrame(summary_data)
        
        # Create top performers DataFrame
        f1_array = np.array(threshold_results['f1_score'])
        top10_indices = np.argsort(f1_array)[-10:][::-1]  # Top 10 F1 scores
        
        top_performers_data = {
            'Rank': list(range(1, 11)),
            'Threshold': [threshold_results['thresholds'][i] for i in top10_indices],
            'F1_Score': [f1_array[i] for i in top10_indices],
            'Precision': [threshold_results['precision'][i] for i in top10_indices],
            'Recall': [threshold_results['recall'][i] for i in top10_indices],
            'Accuracy': [threshold_results['accuracy'][i] for i in top10_indices],
            'True_Positives': [threshold_results['true_positives'][i] for i in top10_indices],
            'False_Positives': [threshold_results['false_positives'][i] for i in top10_indices],
            'False_Negatives': [threshold_results['false_negatives'][i] for i in top10_indices],
            'True_Negatives': [threshold_results['true_negatives'][i] for i in top10_indices]
        }
        df_top_performers = pd.DataFrame(top_performers_data)
        
        # Create detailed statistics DataFrame
        stats_data = {
            'Metric': ['Mean_F1_Score', 'Std_F1_Score', 'Max_F1_Score', 'Min_F1_Score',
                      'Mean_Precision', 'Std_Precision', 'Max_Precision', 'Min_Precision',
                      'Mean_Recall', 'Std_Recall', 'Max_Recall', 'Min_Recall',
                      'Mean_Accuracy', 'Std_Accuracy', 'Max_Accuracy', 'Min_Accuracy'],
            'Value': [
                np.mean(threshold_results['f1_score']),
                np.std(threshold_results['f1_score']),
                np.max(threshold_results['f1_score']),
                np.min(threshold_results['f1_score']),
                np.mean(threshold_results['precision']),
                np.std(threshold_results['precision']),
                np.max(threshold_results['precision']),
                np.min(threshold_results['precision']),
                np.mean(threshold_results['recall']),
                np.std(threshold_results['recall']),
                np.max(threshold_results['recall']),
                np.min(threshold_results['recall']),
                np.mean(threshold_results['accuracy']),
                np.std(threshold_results['accuracy']),
                np.max(threshold_results['accuracy']),
                np.min(threshold_results['accuracy'])
            ]
        }
        df_stats = pd.DataFrame(stats_data)
        
        # Save to Excel with multiple sheets - append if file exists
        file_exists = os.path.exists(save_path)
        
        if file_exists:
            # Load existing file to preserve other model results
            with pd.ExcelWriter(save_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                # Add new model results with timestamped sheet name
                df_results.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"Appended results to existing Excel file: {save_path}")
        else:
            # Create new file with all sheets
            with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
                # Sheet 1: All Results for this model
                df_results.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Sheet 2: Summary for this model
                df_summary.to_excel(writer, sheet_name=f'{sheet_name}_Summary', index=False)
                
                # Sheet 3: Top 10 Performers for this model
                df_top_performers.to_excel(writer, sheet_name=f'{sheet_name}_Top10', index=False)
                
                # Sheet 4: Statistics for this model
                df_stats.to_excel(writer, sheet_name=f'{sheet_name}_Stats', index=False)
                
                # Sheet 5: Threshold Analysis Info for this model
                info_data = {
                    'Parameter': ['Model_Timestamp', 'Analysis_Method', 'Point_Adjustment_Used', 'Total_Thresholds_Tested',
                                 'Evaluation_Metrics', 'Best_Threshold_Found',
                                 'Best_F1_Score_Achieved', 'Best_Precision_Achieved', 'Best_Recall_Achieved'],
                    'Value': [model_timestamp, 'Comprehensive Threshold Analysis', 'Yes', len(threshold_results['thresholds']),
                             'Precision, Recall, F1-Score, Accuracy, TP, FP, FN, TN',
                             f"{threshold_results['best_threshold']:.6f}",
                             f"{best_metrics['f1']:.4f}",
                             f"{best_metrics['precision']:.4f}",
                             f"{best_metrics['recall']:.4f}"]
                }
                df_info = pd.DataFrame(info_data)
                df_info.to_excel(writer, sheet_name=f'{sheet_name}_Info', index=False)
                
                # Sheet 6: Configuration Information for this model
                if config_info is not None:
                    config_data = {
                        'Parameter': list(config_info.keys()),
                        'Value': [str(config_info[key]) for key in config_info.keys()]
                    }
                    df_config = pd.DataFrame(config_data)
                    df_config.to_excel(writer, sheet_name=f'{sheet_name}_Config', index=False)
                
                print(f"Created new Excel file: {save_path}")
        
        # Determine sheet count
        sheet_count = 6 if config_info is not None else 5
        
        print(f"Excel file saved successfully with {sheet_count} sheets:")
        print(f"  1. {sheet_name}: Complete threshold analysis results for this model")
        print(f"  2. {sheet_name}_Summary: Best performance summary")
        print(f"  3. {sheet_name}_Top10: Top 10 F1-score results")
        print(f"  4. {sheet_name}_Stats: Statistical analysis of all metrics")
        print(f"  5. {sheet_name}_Info: Analysis parameters and information")
        if config_info is not None:
            print(f"  6. {sheet_name}_Config: Model and inference configuration")
            print(f"     - Includes: Use_Contrastive, Best_Training_Loss, model architecture")
            print(f"     - Training params: learning rate, epochs, batch size, etc.")
            print(f"     - Model params: dimensions, layers, weights, etc.")
    
    def create_model_comparison_sheet(self, excel_path: str):
        """
        Create a comparison sheet that summarizes all models in the Excel file
        """
        try:
            import pandas as pd
            
            # Read all sheets to find model summaries
            excel_file = pd.ExcelFile(excel_path)
            model_summaries = []
            
            for sheet_name in excel_file.sheet_names:
                if sheet_name.endswith('_Summary'):
                    try:
                        df = pd.read_excel(excel_path, sheet_name=sheet_name)
                        # Extract model info
                        model_info = {}
                        for _, row in df.iterrows():
                            metric = row['Metric']
                            value = row['Value']
                            if metric == 'Model_Timestamp':
                                model_info['Model_Timestamp'] = value
                            elif metric.startswith('Best_'):
                                model_info[metric] = value
                        
                        if model_info:
                            model_summaries.append(model_info)
                    except Exception as e:
                        print(f"Warning: Could not read sheet {sheet_name}: {e}")
            
            if len(model_summaries) > 1:
                # Create comparison DataFrame
                comparison_df = pd.DataFrame(model_summaries)
                comparison_df = comparison_df.sort_values('Best_F1_Score', ascending=False)
                
                # Save comparison sheet
                with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                    comparison_df.to_excel(writer, sheet_name='Model_Comparison', index=False)
                
                print(f"Created Model_Comparison sheet with {len(model_summaries)} models")
                print("Models ranked by F1-Score:")
                for i, (_, row) in enumerate(comparison_df.iterrows(), 1):
                    print(f"  {i}. {row['Model_Timestamp']}: F1={row['Best_F1_Score']:.4f}, "
                          f"Precision={row['Best_Precision']:.4f}, Recall={row['Best_Recall']:.4f}")
            
        except Exception as e:
            print(f"Warning: Could not create model comparison sheet: {e}")
    
    def plot_best_threshold_results(self, data: np.ndarray, reconstruction: np.ndarray,
                                   labels: np.ndarray, timestep_scores: np.ndarray,
                                   best_threshold: float, best_metrics: Dict,
                                   save_path: str = None):
        """
        Plot detailed results using the best threshold found
        
        Args:
            data: Original data
            reconstruction: Reconstructed data
            labels: Ground truth labels
            timestep_scores: Timestep-level anomaly scores
            best_threshold: Best threshold found from analysis
            best_metrics: Best metrics dictionary
            save_path: Path to save plot
        """
        # DISABLED: No plotting for automation
        return
        if labels is None or len(labels) == 0:
            print("Cannot plot: No ground truth labels available")
            return
        
        features, time_steps = data.shape
        time_axis = np.arange(time_steps)
        
        # Detect anomalies using best threshold
        timestep_anomalies = timestep_scores > best_threshold
        
        # Create comprehensive figure
        # fig, axes = plt.subplots(4, 1, figsize=(20, 16))  # DISABLED: No plotting for automation
        
        # Plot 1: Original vs Reconstructed Data (Feature 1)
        ax1 = axes[0]
        ax1.plot(time_axis, data[0], 'b-', label='Original Feature 1', alpha=0.8, linewidth=1.5)
        ax1.plot(time_axis, reconstruction[0], 'g-', label='Reconstructed Feature 1', alpha=0.8, linewidth=1.5)
        
        # Shade ground-truth anomaly regions
        is_anom = (labels == 1).astype(int)
        if np.any(is_anom):
            diff = np.diff(np.concatenate(([0], is_anom, [0])))
            starts = np.nonzero(diff == 1)[0]
            ends = np.nonzero(diff == -1)[0] - 1
            for start_idx, end_idx in zip(starts, ends):
                ax1.axvspan(start_idx, end_idx + 1, alpha=0.25, color='red', 
                           label='GT Anomaly' if start_idx == starts[0] else "")
        
        # Highlight detected anomalies
        if np.any(timestep_anomalies):
            anomaly_indices = np.nonzero(timestep_anomalies)[0]
            ax1.scatter(anomaly_indices, data[0, anomaly_indices],
                       c='orange', s=15, alpha=0.8, label='Detected Anomalies', marker='o', zorder=5)
        
        ax1.set_title(f'Feature 1 - Original vs Reconstructed (Best Threshold: {best_threshold:.6f})', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Amplitude')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Original vs Reconstructed Data (Feature 2)
        ax2 = axes[1]
        ax2.plot(time_axis, data[1], 'b-', label='Original Feature 2', alpha=0.8, linewidth=1.5)
        ax2.plot(time_axis, reconstruction[1], 'g-', label='Reconstructed Feature 2', alpha=0.8, linewidth=1.5)
        
        # Shade ground-truth anomaly regions
        for start_idx, end_idx in zip(starts, ends):
            ax2.axvspan(start_idx, end_idx + 1, alpha=0.25, color='red')
        
        # Highlight detected anomalies
        if np.any(timestep_anomalies):
            ax2.scatter(anomaly_indices, data[1, anomaly_indices],
                       c='orange', s=15, alpha=0.8, label='Detected Anomalies', marker='o', zorder=5)
        
        ax2.set_title(f'Feature 2 - Original vs Reconstructed (Best Threshold: {best_threshold:.6f})', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Amplitude')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Anomaly Scores with Best Threshold
        ax3 = axes[2]
        ax3.plot(time_axis, timestep_scores, 'purple', label='Anomaly Scores', linewidth=1.5)
        ax3.axhline(y=best_threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Best Threshold ({best_threshold:.6f})')
        
        # Highlight detected anomalies on scores plot
        if np.any(timestep_anomalies):
            ax3.scatter(anomaly_indices, timestep_scores[anomaly_indices], 
                       c='red', s=20, alpha=0.8, label='Anomalous Points', zorder=5)
        
        ax3.set_title('Anomaly Scores with Best Threshold', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Anomaly Score')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Performance Metrics Summary
        ax4 = axes[3]
        ax4.axis('off')
        
        # Create performance summary text
        metrics_text = f"""
        BEST THRESHOLD PERFORMANCE SUMMARY
        
        Threshold Value: {best_threshold:.6f}
        
        Performance Metrics:
        • F1-Score: {best_metrics['f1']:.4f}
        • Precision: {best_metrics['precision']:.4f}
        • Recall: {best_metrics['recall']:.4f}
        • Accuracy: {best_metrics['accuracy']:.4f}
        
        Confusion Matrix:
        • True Positives: {best_metrics['tp']}
        • False Positives: {best_metrics['fp']}
        • False Negatives: {best_metrics['fn']}
        • True Negatives: {best_metrics['tn']}
        
        Detection Statistics:
        • Total Timesteps: {len(labels)}
        • Ground Truth Anomalies: {np.sum(labels == 1)} ({np.sum(labels == 1)/len(labels)*100:.1f}%)
        • Detected Anomalies: {np.sum(timestep_anomalies)} ({np.sum(timestep_anomalies)/len(labels)*100:.1f}%)
        • Score Range: [{np.min(timestep_scores):.6f}, {np.max(timestep_scores):.6f}]
        """
        
        ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        plt.suptitle(f'Anomaly Detection Results - Best Threshold Analysis\n'
                    f'F1-Score: {best_metrics["f1"]:.4f} | '
                    f'Precision: {best_metrics["precision"]:.4f} | '
                    f'Recall: {best_metrics["recall"]:.4f}', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Best threshold results plot saved to {save_path}")
        
        plt.show()
        
        # Print detailed summary
        print(f"\n" + "="*80)
        print(f"BEST THRESHOLD RESULTS SUMMARY")
        print(f"="*80)
        print(f"Best Threshold: {best_threshold:.6f}")
        print(f"F1-Score: {best_metrics['f1']:.4f}")
        print(f"Precision: {best_metrics['precision']:.4f}")
        print(f"Recall: {best_metrics['recall']:.4f}")
        print(f"Accuracy: {best_metrics['accuracy']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  True Positives: {best_metrics['tp']}")
        print(f"  False Positives: {best_metrics['fp']}")
        print(f"  False Negatives: {best_metrics['fn']}")
        print(f"  True Negatives: {best_metrics['tn']}")
        print(f"\nDetection Summary:")
        print(f"  Ground Truth Anomalies: {np.sum(labels == 1)} ({np.sum(labels == 1)/len(labels)*100:.1f}%)")
        print(f"  Detected Anomalies: {np.sum(timestep_anomalies)} ({np.sum(timestep_anomalies)/len(labels)*100:.1f}%)")
        print(f"  Anomaly Score Range: [{np.min(timestep_scores):.6f}, {np.max(timestep_scores):.6f}]")
        print(f"="*80)
    
    def plot_results(self, data: np.ndarray, reconstruction: np.ndarray, 
                    labels: np.ndarray, anomaly_scores: np.ndarray, 
                    anomalies: np.ndarray, window_size: int, stride: int = 1,
                    threshold_value: float = None,
                    save_path: str = None,
                    timestep_scores: np.ndarray = None,
                    timestep_anomalies: np.ndarray = None):
        """
        Plot original data, reconstruction, and anomalies on a single grid
        
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
        # DISABLED: No plotting for automation
        return
        features, time_steps = data.shape
        time_axis = np.arange(time_steps)

        # Create figure: feature plots + anomaly scores (no separate labels grid)
        fig, axes = plt.subplots(features + 1, 1, figsize=(15, 3 * (features + 1)))
        
        # Plot each feature
        for i in range(features):
            ax = axes[i]

            # Plot original data
            ax.plot(time_axis, data[i], 'b-', label='Original', alpha=0.7, linewidth=1)
            
            # Plot reconstruction
            ax.plot(time_axis, reconstruction[i], 'g-', label='Reconstruction', alpha=0.7, linewidth=1)
            
            # Shade ground-truth anomaly regions (label==1) directly on the feature plot
            if labels is not None and len(labels) == time_steps:
                # Compute contiguous regions where labels==1
                is_anom = (labels == 1).astype(int)
                if np.any(is_anom):
                    diff = np.diff(np.concatenate(([0], is_anom, [0])))
                    starts = np.nonzero(diff == 1)[0]
                    ends = np.nonzero(diff == -1)[0] - 1
                    for start_idx, end_idx in zip(starts, ends):
                        ax.axvspan(start_idx, end_idx + 1, alpha=0.25, color='red', 
                                   label='GT Anomaly' if start_idx == starts[0] else "")
            
            # Highlight detected anomaly points at timestep resolution if provided
            if timestep_anomalies is not None and np.any(timestep_anomalies):
                anomaly_indices = np.nonzero(timestep_anomalies)[0]
                ax.scatter(anomaly_indices, data[i, anomaly_indices],
                           c='orange', s=8, alpha=0.7, label='Detected Anomaly Points', marker='o', zorder=4)
            
            ax.set_title(f'Feature {i+1} - ECG Signal')
            ax.set_ylabel('Amplitude')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        
        # Plot anomaly scores
        ax = axes[features]
        window_indices = np.arange(len(anomaly_scores))
        ax.plot(window_indices, anomaly_scores, 'purple', label='Window Anomaly Scores', linewidth=1)
        
        # Add threshold line (value can be injected by caller)
        if 'threshold_value' in locals() and threshold_value is not None:
            thr = float(threshold_value)
            label_text = f'Threshold ({thr:.4f})'
        else:
            thr = np.percentile(anomaly_scores, 95)
            label_text = 'Threshold (95%)'
        ax.axhline(y=thr, color='red', linestyle='--', alpha=0.7, label=label_text)
        
        # Highlight detected anomalies
        if np.any(anomalies):
            ax.scatter(window_indices[anomalies], anomaly_scores[anomalies], 
                      c='red', s=20, alpha=0.8, label='Anomalous Windows', zorder=5)
        
        ax.set_title('Anomaly Scores')
        ax.set_ylabel('Score')
        ax.set_xlabel('Window Index')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add basic statistics box on the scores plot if labels exist
        if labels is not None and len(labels) == time_steps:
            anomaly_count = int(np.sum(labels == 1))
            total_count = int(len(labels))
            anomaly_ratio = anomaly_count / max(total_count, 1) * 100
            ax.text(0.02, 0.95, f'GT anomalies: {anomaly_count}/{total_count} ({anomaly_ratio:.1f}%)',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.8})
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_threshold_analysis(self, threshold_results: Dict, save_path: str = None):
        """
        Plot comprehensive threshold analysis results showing performance metrics across different thresholds
        
        Args:
            threshold_results: Results from evaluate_threshold_range
            save_path: Path to save plot
        """
        # DISABLED: No plotting for automation
        return
        if "error" in threshold_results:
            print(f"Cannot plot: {threshold_results['error']}")
            return
        
        thresholds = threshold_results['thresholds']
        precision = threshold_results['precision']
        recall = threshold_results['recall']
        f1_scores = threshold_results['f1_score']
        accuracy = threshold_results['accuracy']
        tp = threshold_results['true_positives']
        fp = threshold_results['false_positives']
        fn = threshold_results['false_negatives']
        tn = threshold_results['true_negatives']
        
        # Create a comprehensive plot with 6 subplots
        fig, axes = plt.subplots(3, 2, figsize=(18, 15))
        
        # Plot 1: Precision vs Threshold
        axes[0, 0].plot(thresholds, precision, 'b-', linewidth=2, label='Precision')
        axes[0, 0].axvline(x=threshold_results['best_threshold'], color='red', linestyle='--', 
                          label=f'Best Threshold ({threshold_results["best_threshold"]:.4f})')
        axes[0, 0].set_xlabel('Threshold')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].set_title('Precision vs Threshold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Recall vs Threshold
        axes[0, 1].plot(thresholds, recall, 'g-', linewidth=2, label='Recall')
        axes[0, 1].axvline(x=threshold_results['best_threshold'], color='red', linestyle='--', 
                          label=f'Best Threshold ({threshold_results["best_threshold"]:.4f})')
        axes[0, 1].set_xlabel('Threshold')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].set_title('Recall vs Threshold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: F1-Score vs Threshold
        axes[1, 0].plot(thresholds, f1_scores, 'purple', linewidth=2, label='F1-Score')
        axes[1, 0].axvline(x=threshold_results['best_threshold'], color='red', linestyle='--', 
                          label=f'Best Threshold ({threshold_results["best_threshold"]:.4f})')
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].set_title('F1-Score vs Threshold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Accuracy vs Threshold
        axes[1, 1].plot(thresholds, accuracy, 'orange', linewidth=2, label='Accuracy')
        axes[1, 1].axvline(x=threshold_results['best_threshold'], color='red', linestyle='--', 
                          label=f'Best Threshold ({threshold_results["best_threshold"]:.4f})')
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Accuracy vs Threshold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 5: Confusion Matrix Components vs Threshold
        axes[2, 0].plot(thresholds, tp, 'g-', linewidth=2, label='True Positives')
        axes[2, 0].plot(thresholds, fp, 'r-', linewidth=2, label='False Positives')
        axes[2, 0].axvline(x=threshold_results['best_threshold'], color='red', linestyle='--', alpha=0.7)
        axes[2, 0].set_xlabel('Threshold')
        axes[2, 0].set_ylabel('Count')
        axes[2, 0].set_title('TP/FP vs Threshold')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # Plot 6: FN/TN vs Threshold
        axes[2, 1].plot(thresholds, fn, 'orange', linewidth=2, label='False Negatives')
        axes[2, 1].plot(thresholds, tn, 'blue', linewidth=2, label='True Negatives')
        axes[2, 1].axvline(x=threshold_results['best_threshold'], color='red', linestyle='--', alpha=0.7)
        axes[2, 1].set_xlabel('Threshold')
        axes[2, 1].set_ylabel('Count')
        axes[2, 1].set_title('FN/TN vs Threshold')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        # Add comprehensive title with best results
        best_metrics = threshold_results['best_metrics']
        title = f'Threshold Analysis Results (Best F1: {threshold_results["best_f1"]:.4f} @ {threshold_results["best_threshold"]:.4f})'
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Threshold analysis plot saved to {save_path}")
        
        plt.show()
        
        # Print detailed statistics
        print(f"\n" + "="*80)
        print(f"DETAILED THRESHOLD ANALYSIS STATISTICS")
        print(f"="*80)
        print(f"Total thresholds evaluated: {len(thresholds)}")
        print(f"\nBest Performance:")
        print(f"  Threshold: {threshold_results['best_threshold']:.6f}")
        print(f"  F1-Score: {best_metrics['f1']:.4f}")
        print(f"  Precision: {best_metrics['precision']:.4f}")
        print(f"  Recall: {best_metrics['recall']:.4f}")
        print(f"  Accuracy: {best_metrics['accuracy']:.4f}")
        print(f"  True Positives: {best_metrics['tp']}")
        print(f"  False Positives: {best_metrics['fp']}")
        print(f"  False Negatives: {best_metrics['fn']}")
        print(f"  True Negatives: {best_metrics['tn']}")
        
        # Find top 5 F1 scores
        f1_array = np.array(f1_scores)
        top5_indices = np.argsort(f1_array)[-5:][::-1]
        print(f"\nTop 5 F1-Scores:")
        for i, idx in enumerate(top5_indices):
            print(f"  {i+1}. F1: {f1_array[idx]:.4f}, Threshold: {thresholds[idx]:.6f}")
        print(f"="*80)
    
    def evaluate_performance(self, labels: np.ndarray, anomalies: np.ndarray, 
                           window_size: int, stride: int = 1, use_adjustment: bool = True) -> Dict[str, float]:
        """
        Evaluate anomaly detection performance with optional Point Adjustment
        
        Args:
            labels: Ground truth labels
            anomalies: Detected anomalies
            window_size: Window size
            stride: Stride
            use_adjustment: Whether to apply Point Adjustment algorithm
            
        Returns:
            Dictionary of performance metrics
        """
        if labels is None or len(labels) == 0:
            return {"error": "No ground truth labels available"}
        
        # Convert window-level anomalies to time-step level
        time_steps = len(labels)
        
        detected_timesteps = np.zeros(time_steps, dtype=bool)
        for i, is_anomaly in enumerate(anomalies):
            if is_anomaly:
                start_idx = i * stride
                end_idx = min(start_idx + window_size, time_steps)
                detected_timesteps[start_idx:end_idx] = True
        
        # Convert to binary arrays
        gt = labels.astype(int)
        pred_before_pa = detected_timesteps.astype(int)
        
        # Apply Point Adjustment if requested
        if use_adjustment:
            _, pred_after_pa = adjustment(gt.copy(), pred_before_pa.copy())
        else:
            pred_after_pa = pred_before_pa
        
        # Calculate metrics after Point Adjustment
        acc, precision, recall, f1, (tp, fp, fn, tn) = binary_classification_metrics(gt, pred_after_pa)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": acc,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn
        }


def calculate_average_metrics(all_threshold_results: List[Dict]) -> Dict:
    """
    Calculate basic metrics across all files
    
    Args:
        all_threshold_results: List of threshold results for each file
    
    Returns:
        Dictionary with basic metrics
    """
    if not all_threshold_results:
        return {}
    
    # Return only basic information
    avg_results = {
        'num_files': len(all_threshold_results)
    }
    
    return avg_results


def save_comprehensive_results_to_excel(all_threshold_results: List[Dict], avg_results: Dict, 
                                      save_path: str, args):
    """
    Save comprehensive results to Excel with multiple sheets
    
    Args:
        all_threshold_results: List of threshold results for each file
        avg_results: Average metrics across all files
        save_path: Path to save Excel file
        args: Command line arguments
    """
    print(f"Saving comprehensive results to Excel: {save_path}")
    
    # Load additional information from config and checkpoint
    additional_info = {}
    
    # Load config information
    config_path = os.path.join(os.path.dirname(args.model_path), 'config.json')
    if os.path.exists(config_path):
        try:
            import json
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                additional_info.update({
                    'Use_Contrastive': config_data.get('use_contrastive', 'Unknown'),
                    'Contrastive_Weight': config_data.get('contrastive_weight', 'Unknown'),
                    'Reconstruction_Weight': config_data.get('reconstruction_weight', 'Unknown'),
                    'Temperature': config_data.get('temperature', 'Unknown'),
                    'D_Model': config_data.get('d_model', 'Unknown'),
                    'Projection_Dim': config_data.get('projection_dim', 'Unknown'),
                    'Transformer_Layers': config_data.get('transformer_layers', 'Unknown'),
                    'TCN_Layers': config_data.get('tcn_num_layers', 'Unknown'),
                    'Learning_Rate': config_data.get('learning_rate', 'Unknown'),
                    'Weight_Decay': config_data.get('weight_decay', 'Unknown'),
                    'Epsilon': config_data.get('epsilon', 'Unknown'),
                    'Training_Batch_Size': config_data.get('batch_size', 'Unknown'),
                    'Num_Epochs': config_data.get('num_epochs', 'Unknown'),
                    'Training_Device': config_data.get('device', 'Unknown'),
                    'Seed': config_data.get('seed', 'Unknown'),
                    'Mask_Mode': config_data.get('mask_mode', 'Unknown'),
                    'Mask_Ratio': config_data.get('mask_ratio', 'Unknown'),
                    'Mask_Seed': config_data.get('mask_seed', 'Unknown'),
                    # Augmentation-specific hyperparameters
                    'Aug_nhead': config_data.get('aug_nhead', 'Unknown'),
                    'Aug_num_layers': config_data.get('aug_num_layers', 'Unknown'),
                    'Aug_tcn_kernel_size': config_data.get('aug_tcn_kernel_size', 'Unknown'),
                    'Aug_tcn_num_layers': config_data.get('aug_tcn_num_layers', 'Unknown'),
                    'Aug_dropout': config_data.get('aug_dropout', 'Unknown'),
                    'Aug_temperature': config_data.get('aug_temperature', 'Unknown')
                })
        except Exception as e:
            print(f"Warning: Could not load config.json: {e}")
    
    # Load checkpoint information for best loss
    try:
        import torch
        checkpoint = torch.load(args.model_path, map_location='cpu')
        additional_info.update({
            'Best_Training_Loss': checkpoint.get('best_loss', 'Unknown'),
            'Training_Epoch': checkpoint.get('epoch', 'Unknown'),
            'Has_Optimizer': checkpoint.get('optimizer_state_dict') is not None,
            'Has_Scheduler': checkpoint.get('scheduler_state_dict') is not None,
            'Contrastive_Weight_Used': checkpoint.get('contrastive_weight', 'Unknown'),
            'Reconstruction_Weight_Used': checkpoint.get('reconstruction_weight', 'Unknown')
        })
    except Exception as e:
        print(f"Warning: Could not load checkpoint info: {e}")
    
    # Prepare comprehensive config information
    config_info = {
        'Dataset_Type': args.dataset,
        'Data_Path': args.data_path,
        'Model_Path': args.model_path,
        'Window_Size': args.window_size,
        'Stride': args.stride,
        'Inference_Batch_Size': args.batch_size,
        'Num_Thresholds': args.num_thresholds,
        'Use_Threshold_Range': True,
        'Mask_Mode': args.mask_mode,
        'Mask_Ratio': args.mask_ratio,
        'Mask_Seed': args.mask_seed,
        'Save_Plot': args.save_plot,
        'Save_Excel': args.save_excel,
        'Output_Directory': args.output_dir,
        'Input_Dimension': 2,  # ECG has 2 features
        'Model_Architecture': 'ContrastiveModel',
        'Analysis_Type': 'Per-File Analysis',
        'Total_Files_Processed': len(all_threshold_results)
    }
    
    # Merge additional info
    config_info.update(additional_info)
    
    file_exists = os.path.exists(save_path)
    
    # Build current run frames
    summary_data = []
    for result in all_threshold_results:
        filename = result.get('filename', 'Unknown')
        best_metrics = result.get('best_metrics', {})
        summary_data.append({
            'Filename': filename,
            'F1_Score': best_metrics.get('f1', 0.0),
            'Precision': best_metrics.get('precision', 0.0),
            'Recall': best_metrics.get('recall', 0.0),
            'Accuracy': best_metrics.get('accuracy', 0.0),
            'Best_Threshold': result.get('best_threshold', 0.0),
            'Total_Samples': result.get('total_samples', 0),
            'Total_Anomalies_GT': result.get('total_anomalies', 0),
            'Detected_Anomalies': result.get('detected_anomalies', 0),
            'Detection_Rate': result.get('anomaly_detection_rate', 0.0),
            'TP': best_metrics.get('tp', 0),
            'FP': best_metrics.get('fp', 0),
            'FN': best_metrics.get('fn', 0),
            'TN': best_metrics.get('tn', 0),
            'Data_Shape': str(result.get('data_shape', 'Unknown'))
        })
    df_summary_new = pd.DataFrame(summary_data)
    df_summary_new = df_summary_new.sort_values('F1_Score', ascending=False)

    anomaly_stats = []
    total_gt_anomalies = 0
    total_detected_anomalies = 0
    total_samples = 0
    for result in all_threshold_results:
        filename = result.get('filename', 'Unknown')
        gt_anomalies = result.get('total_anomalies', 0)
        detected_anomalies = result.get('detected_anomalies', 0)
        samples = result.get('total_samples', 0)
        total_gt_anomalies += gt_anomalies
        total_detected_anomalies += detected_anomalies
        total_samples += samples
        anomaly_stats.append({
            'Filename': filename,
            'Total_Samples': samples,
            'Ground_Truth_Anomalies': gt_anomalies,
            'Detected_Anomalies': detected_anomalies,
            'Detection_Rate': result.get('anomaly_detection_rate', 0.0),
            'GT_Anomaly_Ratio': gt_anomalies / samples if samples > 0 else 0.0,
            'Detection_Accuracy': detected_anomalies / gt_anomalies if gt_anomalies > 0 else 0.0
        })
    df_anomaly_new = pd.DataFrame(anomaly_stats)

    if not file_exists:
        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            df_summary_new.to_excel(writer, sheet_name='Per_File_Summary', index=False)
            df_anomaly_new.to_excel(writer, sheet_name='Anomaly_Detection_Stats', index=False)
            overall_stats = {
                'Metric': [
                    'Total_Files_Processed',
                    'Total_Samples_Across_All_Files',
                    'Total_GT_Anomalies',
                    'Total_Detected_Anomalies',
                    'Overall_Detection_Rate',
                    'Overall_GT_Anomaly_Ratio',
                    'Overall_Detection_Accuracy'
                ],
                'Value': [
                    len(all_threshold_results),
                    total_samples,
                    total_gt_anomalies,
                    total_detected_anomalies,
                    total_detected_anomalies / total_samples if total_samples > 0 else 0.0,
                    total_gt_anomalies / total_samples if total_samples > 0 else 0.0,
                    total_detected_anomalies / total_gt_anomalies if total_gt_anomalies > 0 else 0.0
                ]
            }
            pd.DataFrame(overall_stats).to_excel(writer, sheet_name='Overall_Anomaly_Stats', index=False)
            df_summary_new.head(5).to_excel(writer, sheet_name='Top_5_Performers', index=False)
            cfg_df = pd.DataFrame({'Parameter': list(config_info.keys()), 'Value': [str(config_info[k]) for k in config_info.keys()]})
            cfg_df.to_excel(writer, sheet_name='Configuration', index=False)
            if all_threshold_results and 'all_results' in all_threshold_results[0]:
                all_detailed = []
                for result in all_threshold_results:
                    filename = result.get('filename', 'Unknown')
                    detailed_results = result.get('all_results', [])
                    for row in detailed_results:
                        row['filename'] = filename
                        all_detailed.append(row)
                if all_detailed:
                    pd.DataFrame(all_detailed).to_excel(writer, sheet_name='Detailed_Results', index=False)
    else:
        existing = pd.ExcelFile(save_path)
        with pd.ExcelWriter(save_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            # Per_File_Summary
            if 'Per_File_Summary' in existing.sheet_names:
                df_old = pd.read_excel(save_path, sheet_name='Per_File_Summary')
                df_concat = pd.concat([df_old, df_summary_new], ignore_index=True)
            else:
                df_concat = df_summary_new
            df_concat = df_concat.sort_values('F1_Score', ascending=False)
            df_concat.to_excel(writer, sheet_name='Per_File_Summary', index=False)

            # Anomaly_Detection_Stats
            if 'Anomaly_Detection_Stats' in existing.sheet_names:
                df_old = pd.read_excel(save_path, sheet_name='Anomaly_Detection_Stats')
                df_anom_concat = pd.concat([df_old, df_anomaly_new], ignore_index=True)
            else:
                df_anom_concat = df_anomaly_new
            df_anom_concat.to_excel(writer, sheet_name='Anomaly_Detection_Stats', index=False)

            # Overall stats recomputed from summary
            df_all = df_concat
            total_samples_all = int(df_all['Total_Samples'].sum()) if 'Total_Samples' in df_all.columns else 0
            total_gt_all = int(df_all['Total_Anomalies_GT'].sum()) if 'Total_Anomalies_GT' in df_all.columns else 0
            total_det_all = int(df_all['Detected_Anomalies'].sum()) if 'Detected_Anomalies' in df_all.columns else 0
            overall_stats = {
                'Metric': [
                    'Total_Files_Processed',
                    'Total_Samples_Across_All_Files',
                    'Total_GT_Anomalies',
                    'Total_Detected_Anomalies',
                    'Overall_Detection_Rate',
                    'Overall_GT_Anomaly_Ratio',
                    'Overall_Detection_Accuracy'
                ],
                'Value': [
                    len(df_all),
                    total_samples_all,
                    total_gt_all,
                    total_det_all,
                    (total_det_all / total_samples_all) if total_samples_all > 0 else 0.0,
                    (total_gt_all / total_samples_all) if total_samples_all > 0 else 0.0,
                    (total_det_all / total_gt_all) if total_gt_all > 0 else 0.0
                ]
            }
            pd.DataFrame(overall_stats).to_excel(writer, sheet_name='Overall_Anomaly_Stats', index=False)

            # Top performers from combined summary
            df_concat.head(5).to_excel(writer, sheet_name='Top_5_Performers', index=False)

            # Configuration: append rows
            cfg_df_new = pd.DataFrame({'Parameter': list(config_info.keys()), 'Value': [str(config_info[k]) for k in config_info.keys()]})
            if 'Configuration' in existing.sheet_names:
                cfg_old = pd.read_excel(save_path, sheet_name='Configuration')
                cfg_concat = pd.concat([cfg_old, cfg_df_new], ignore_index=True)
            else:
                cfg_concat = cfg_df_new
            cfg_concat.to_excel(writer, sheet_name='Configuration', index=False)
    
    print(f"Comprehensive Excel file saved successfully with multiple sheets:")
    print(f"  1. Per_File_Summary: Results for each individual file")
    print(f"  2. Anomaly_Detection_Stats: Detailed anomaly detection per file")
    print(f"  3. Overall_Anomaly_Stats: Overall anomaly detection statistics")
    print(f"  4. Top_5_Performers: Best performing files")
    print(f"  5. Configuration: Model and inference configuration")
    print(f"  6. Detailed_Results: Complete threshold analysis results")


def main():
    """Main function for inference"""
    parser = argparse.ArgumentParser(description='Contrastive Learning Model Inference')
    
    # Model arguments
    parser.add_argument('--dataset', type=str, default='ucr',
                       choices=['ecg', 'psm', 'nab', 'smap_msl', 'smd'],
                       help='Type of dataset')
    parser.add_argument('--data_path', type=str, default=r'D:/Hoc_voi_cha_hanh/FPT/Hoc_rieng/ICIIT2025/MainModel/datasets/ucr/labeled',
                       help='Path to test data')
    parser.add_argument('--model_path', type=str, default='D:/Hoc_voi_cha_hanh/FPT/Hoc_rieng/ICIIT2025/MainModel/checkpoints_ucr_max/ucr_20250930_050118/best_model.pth',
                       help='Path to model checkpoint (if None, will use checkpoints/{dataset}/best_model.pth)')
    # Optional: specific test filename. If not provided, will try to read from config.json next to model_path
    parser.add_argument('--dataset_name', type=str, default=None,
                       help='Specific test filename to run (e.g., chfdb_chf01_275.pkl). If omitted, will try to read dataset_name from config.json')
    # Inference arguments
    parser.add_argument('--window_size', type=int, default=128,
                       help='Window size for sliding window (will be overridden by config if available)')
    parser.add_argument('--stride', type=int, default=1,
                       help='Stride for sliding window')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference (will be overridden by config if available)')
    parser.add_argument('--num_thresholds', type=int, default=1000,
                       help='Number of threshold values to test in range evaluation (500-1000 recommended)')
    # Masking options
    parser.add_argument('--mask_mode', type=str, default='none', choices=['none', 'time', 'feature'],
                       help='Masking mode for augmented input (none, time, feature)')
    parser.add_argument('--mask_ratio', type=float, default=0.0,
                       help='Fraction of timesteps/features to mask (0.0 - 1.0)')
    parser.add_argument('--mask_seed', type=int, default=None,
                       help='Random seed for masking reproducibility')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save results (if None, will use inference_results_{dataset})')
    parser.add_argument('--save_plot', action='store_true',
                       help='Save plot to file')
    parser.add_argument('--save_excel', action='store_true', default=True,
                       help='Save threshold analysis results to Excel file')
    
    args = parser.parse_args()
    
    # Set default model path if not provided
    if args.model_path is None:
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Find latest timestamped checkpoint for this dataset
        checkpoints_base = os.path.join('/kaggle', 'working', 'ICIIT2025', 'checkpoints')
        latest_timestamp = None
        
        if os.path.exists(checkpoints_base):
            try:
                # Get all directories that start with dataset name
                import re
                pattern = f"^{args.dataset}_\\d{{8}}_\\d{{6}}$"
                matching_dirs = []
                
                for item in os.listdir(checkpoints_base):
                    if os.path.isdir(os.path.join(checkpoints_base, item)) and re.match(pattern, item):
                        matching_dirs.append(item)
                
                if matching_dirs:
                    # Sort by timestamp (newest first)
                    matching_dirs.sort(reverse=True)
                    latest_timestamp = matching_dirs[0]
                    print(f"Found latest checkpoint: {latest_timestamp}")
                else:
                    print(f"No timestamped checkpoints found for dataset: {args.dataset}")
            except Exception as e:
                print(f"Error scanning checkpoints: {e}")
        
        if latest_timestamp:
            args.model_path = os.path.join(checkpoints_base, latest_timestamp, 'best_model.pth')
        else:
            # Fallback to old format
            args.model_path = os.path.join(checkpoints_base, args.dataset, 'best_model.pth')
        
        print(f"Using model path: {args.model_path}")
        
        # Check if model file exists
        if not os.path.exists(args.model_path):
            print(f"ERROR: Model file not found at {args.model_path}")
            print(f"Available checkpoints in {checkpoints_base}:")
            if os.path.exists(checkpoints_base):
                for item in os.listdir(checkpoints_base):
                    print(f"  - {item}")
            else:
                print(f"  Checkpoints directory does not exist: {checkpoints_base}")
            return
    
    # Set default output directory if not provided
    if args.output_dir is None:
        args.output_dir = f'inference_results_{args.dataset}'
        print(f"Using default output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize inference
    inference = ContrastiveInference(args.model_path)
    # Configure masking
    inference.mask_mode = args.mask_mode
    inference.mask_ratio = max(0.0, min(1.0, float(args.mask_ratio)))
    inference.mask_seed = args.mask_seed
    
    # Use window_size from config if available, otherwise use command line argument
    if inference.window_size is not None:
        print(f"Using window_size from config: {inference.window_size}")
        args.window_size = inference.window_size
    else:
        print(f"Using window_size from command line: {args.window_size}")
    
    # Use batch_size from config if available, otherwise use command line argument
    if hasattr(inference, 'batch_size') and inference.batch_size is not None:
        print(f"Using batch_size from config: {inference.batch_size}")
        args.batch_size = inference.batch_size
    else:
        print(f"Using batch_size from command line: {args.batch_size}")
    
    # Load test data
    print(f"Loading test data from {args.data_path}")
    
    # Try to read dataset_name from the model's config.json if CLI not provided
    config_dataset_name = None
    try:
        if not args.dataset_name:
            cfg_path = os.path.join(os.path.dirname(args.model_path), 'config.json')
            if os.path.exists(cfg_path):
                import json
                with open(cfg_path, 'r') as _f:
                    _cfg = json.load(_f)
                    config_dataset_name = _cfg.get('dataset_name')
                    if config_dataset_name:
                        print(f"Using dataset_name from config.json: {config_dataset_name}")
    except Exception as _e:
        print(f"Warning: unable to read dataset_name from config.json: {_e}")
    
    if args.dataset == 'ecg':
        # Load ECG data - test each file separately
        test_path = os.path.join(args.data_path, "labeled", "test")
        # Prefer CLI --dataset_name. If absent, fall back to config.json's dataset_name.
        chosen_name = None
        if hasattr(args, 'dataset_name') and args.dataset_name:
            chosen_name = args.dataset_name
        elif config_dataset_name:
            chosen_name = config_dataset_name

        if chosen_name:
            chosen_path = os.path.join(test_path, chosen_name)
            if os.path.exists(chosen_path):
                test_files = [chosen_name]
                print(f"Selected single test file: {chosen_name}")
            else:
                print(f"Specified dataset_name not found in test folder: {chosen_name}")
                test_files = []
        else:
            test_files = [f for f in os.listdir(test_path) if f.endswith('.pkl')]
        
        if not test_files:
            print("No test files found!")
            return
        
        print(f"Found {len(test_files)} test files. Testing each file separately...")
        
        # Store results for each file
        all_file_results = []
        all_threshold_results = []
        
        # Process each test file
        for file_idx, test_file in enumerate(test_files):
            print(f"\n{'='*60}")
            print(f"PROCESSING FILE {file_idx + 1}/{len(test_files)}: {test_file}")
            print(f"{'='*60}")
            
            test_path_full = os.path.join(test_path, test_file)
            
            try:
                with open(test_path_full, 'rb') as f:
                    test_data = pickle.load(f)
                
                if isinstance(test_data, list):
                    test_data = np.array(test_data)
                
                # Ensure correct shape (features, time_steps)
                if test_data.shape[0] > test_data.shape[1]:
                    test_data = test_data.T
                
                # Extract features and labels from ECG data
                # ECG data format: [feature1, feature2, label] where label: 0=normal, 1=anomaly
                if test_data.shape[0] >= 3:  # Has at least 3 rows (2 features + 1 label)
                    # Extract only the 2 features for model input (model is configured with input_dim=2)
                    features = test_data[:2, :]  # First 2 rows are features
                    labels = test_data[2, :]     # Third row is labels (0=normal, 1=anomaly)

                    # Normalize test features using train min/max (same as training)
                    # Try to locate the corresponding train file by name
                    train_file_path = os.path.join(args.data_path, "labeled", "train", test_file)
                    train_min = None
                    train_max = None
                    if os.path.exists(train_file_path):
                        try:
                            with open(train_file_path, 'rb') as f:
                                train_data_raw = pickle.load(f)
                            if isinstance(train_data_raw, list):
                                train_data_raw = np.array(train_data_raw)
                            if train_data_raw.shape[0] > train_data_raw.shape[1]:
                                train_data_raw = train_data_raw.T
                            # Use only the first 2 rows (features) to compute global min/max
                            train_feats = train_data_raw[:2, :]
                            train_min = np.min(train_feats)
                            train_max = np.max(train_feats)
                        except Exception as e:
                            print(f"Warning: failed to load train file for normalization: {e}")

                    test_data = features.astype(np.float32)

                    # Apply global min-max normalization to [0, 1]
                    if train_min is not None and train_max is not None:
                        print("Normalizing test features using train global min/max (as in training)...")
                        denom = (train_max - train_min)
                        if denom > 0:
                            test_data = (test_data - train_min) / denom
                    else:
                        print("Train file not found. Falling back to test-only global min/max normalization...")
                        test_min = np.min(test_data)
                        test_max = np.max(test_data)
                        denom = (test_max - test_min)
                        if denom > 0:
                            test_data = (test_data - test_min) / denom

                    print(f"ECG data loaded: {test_data.shape[0]} features, {test_data.shape[1]} time steps")
                    print(f"Labels shape: {labels.shape}")
                    print(f"Normal samples: {np.sum(labels == 0)} ({np.sum(labels == 0) / len(labels) * 100:.1f}%)")
                    print(f"Anomaly samples: {np.sum(labels == 1)} ({np.sum(labels == 1) / len(labels) * 100:.1f}%)")
                else:
                    # Fallback: no labels available
                    labels = None
                    print("ECG data loaded without labels")
                
                # Run inference on this file
                print(f"\nRunning inference on {test_file}...")
                
                # Sliding window inference (window-level outputs)
                reconstruction_errors, window_anomaly_scores, absolute_errors = inference.sliding_window_inference(
                    test_data, args.window_size, args.stride, args.batch_size
                )

                # Build full-sequence reconstruction from window reconstruction errors
                reconstruction = inference.create_full_reconstruction(
                    test_data, reconstruction_errors, args.window_size, args.stride
                )

                # Convert window absolute errors to per-timestep scores following the required strategy
                timestep_scores = inference.compute_timestep_scores(
                    absolute_errors, test_data.shape[1], args.window_size, args.stride
                )
                
                # Store file-specific results
                file_result = {
                    'filename': test_file,
                    'data_shape': test_data.shape,
                    'reconstruction': reconstruction,
                    'timestep_scores': timestep_scores,
                    'absolute_errors': absolute_errors,
                    'window_anomaly_scores': window_anomaly_scores,
                    'reconstruction_errors': reconstruction_errors,
                    'labels': labels
                }
                all_file_results.append(file_result)
                
                # Ensure labels and timestep_scores have the same length
                if labels is not None and len(labels) != len(timestep_scores):
                    print(f"Info: Labels length ({len(labels)}) != Timestep scores length ({len(timestep_scores)})")
                    print(f"This is normal for sliding window inference. Adjusting labels to match timestep scores length...")
                    
                    # Adjust labels to match timestep_scores length
                    if len(labels) > len(timestep_scores):
                        # Truncate labels to match timestep scores (common case)
                        original_length = len(labels)
                        labels = labels[:len(timestep_scores)]
                        print(f"Truncated labels from {original_length} to {len(labels)}")
                    else:
                        # Pad labels with the last label value (rare case)
                        original_length = len(labels)
                        last_label = labels[-1] if len(labels) > 0 else 0
                        padding = np.full(len(timestep_scores) - len(labels), last_label)
                        labels = np.concatenate([labels, padding])
                        print(f"Padded labels from {original_length} to {len(labels)}")
                    
                    print(f"Final labels length: {len(labels)}")
                
                # Initialize threshold_results
                threshold_results = None
                
                # Run threshold analysis if requested
                # Always use threshold range for ECG data
                if True:
                    print(f"Running threshold analysis for {test_file}...")
                    threshold_results = inference.evaluate_threshold_range(
                        timestep_scores, labels, args.num_thresholds, use_adjustment=True
                    )
                    
                    # Add filename and additional info to threshold results
                    threshold_results['filename'] = test_file
                    threshold_results['data_shape'] = test_data.shape
                    threshold_results['total_samples'] = len(labels) if labels is not None else 0
                    threshold_results['total_anomalies'] = np.sum(labels) if labels is not None else 0
                    
                    # Add detected anomalies info using best threshold
                    if 'best_threshold' in threshold_results and 'best_metrics' in threshold_results:
                        best_threshold = threshold_results['best_threshold']
                        predictions = timestep_scores > best_threshold
                        threshold_results['detected_anomalies'] = np.sum(predictions)
                        threshold_results['anomaly_detection_rate'] = np.sum(predictions) / len(predictions) if len(predictions) > 0 else 0.0
                        
                        # Print file-specific results
                        print(f"  File: {test_file}")
                        print(f"  Total samples: {threshold_results['total_samples']}")
                        print(f"  Total anomalies (GT): {threshold_results['total_anomalies']}")
                        print(f"  Detected anomalies: {threshold_results['detected_anomalies']}")
                        print(f"  Detection rate: {threshold_results['anomaly_detection_rate']:.4f}")
                        print(f"  Best F1-Score: {threshold_results['best_metrics']['f1']:.4f}")
                        print(f"  Best Threshold: {best_threshold:.6f}")
                    
                    all_threshold_results.append(threshold_results)
                    
                    # Plot threshold analysis for this file
                    if args.save_plot:
                        threshold_plot_path = os.path.join(args.output_dir, f'threshold_analysis_{test_file.replace(".pkl", "")}.png')
                        inference.plot_threshold_analysis(threshold_results, threshold_plot_path)
                else:
                    # If not using threshold range, still calculate basic stats
                    if labels is not None:
                        default_threshold = np.percentile(timestep_scores, 95)
                        predictions = timestep_scores > default_threshold
                        basic_stats = {
                            'filename': test_file,
                            'data_shape': test_data.shape,
                            'total_samples': len(labels),
                            'total_anomalies': np.sum(labels),
                            'detected_anomalies': np.sum(predictions),
                            'anomaly_detection_rate': np.sum(predictions) / len(predictions) if len(predictions) > 0 else 0.0,
                            'best_threshold': default_threshold
                        }
                        print(f"  File: {test_file} (basic analysis)")
                        print(f"  Total samples: {basic_stats['total_samples']}")
                        print(f"  Total anomalies (GT): {basic_stats['total_anomalies']}")
                        print(f"  Detected anomalies: {basic_stats['detected_anomalies']}")
                        print(f"  Detection rate: {basic_stats['anomaly_detection_rate']:.4f}")
                        print(f"  Threshold (95th percentile): {default_threshold:.6f}")
                
                # Plot results for this file
                if args.save_plot:
                    best_plot_path = os.path.join(args.output_dir, f'best_threshold_results_{test_file.replace(".pkl", "")}.png')
                    if threshold_results and 'best_threshold' in threshold_results:
                        # Check if best_metrics has the correct keys
                        best_metrics = threshold_results['best_metrics']
                        if 'f1' not in best_metrics:
                            print(f"Warning: best_metrics missing 'f1' key. Available keys: {list(best_metrics.keys())}")
                            # Try to fix if it has 'f1_score' instead
                            if 'f1_score' in best_metrics:
                                best_metrics['f1'] = best_metrics['f1_score']
                                print("Fixed: copied 'f1_score' to 'f1'")
                            else:
                                best_metrics['f1'] = 0.0
                                print("Added default 'f1' = 0.0")
                        
                        inference.plot_best_threshold_results(
                            test_data, reconstruction, labels, timestep_scores,
                            threshold_results['best_threshold'], best_metrics,
                            best_plot_path
                        )
                    else:
                        # Use a default threshold
                        default_threshold = np.percentile(timestep_scores, 95)
                        default_metrics = {
                            'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'accuracy': 0.0
                        }
                        inference.plot_best_threshold_results(
                            test_data, reconstruction, labels, timestep_scores,
                            default_threshold, default_metrics, best_plot_path
                        )
                
                print(f"Completed processing {test_file}")
                
            except Exception as e:
                print(f"Error processing {test_file}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Calculate average results across all files
        print(f"\n{'='*80}")
        print(f"CALCULATING AVERAGE RESULTS ACROSS ALL FILES")
        print(f"{'='*80}")
        
        if all_threshold_results:
            # Calculate basic metrics
            avg_results = calculate_average_metrics(all_threshold_results)
            print(f"Total files processed: {avg_results['num_files']}")
            
            # Save comprehensive results to Excel
            if args.save_excel:
                # Use the same fixed Excel filename for consistency
                # Use distinct filename for single-file vs multi-file runs
                if hasattr(args, 'dataset_name') and args.dataset_name:
                    base = os.path.splitext(os.path.basename(args.dataset_name))[0]
                    excel_path = os.path.join(args.output_dir, f'{args.dataset}_single_{base}_results.xlsx')
                else:
                    excel_path = os.path.join(args.output_dir, f'{args.dataset}_all_models_results.xlsx')
                save_comprehensive_results_to_excel(all_threshold_results, avg_results, excel_path, args)
        
        # Save combined results
        results = {
            'all_file_results': all_file_results,
            'all_threshold_results': all_threshold_results,
            'test_files': test_files,
            'window_size': args.window_size,
            'stride': args.stride,
            'batch_size': args.batch_size,
            'num_thresholds': args.num_thresholds
        }
        
        results_path = os.path.join(args.output_dir, 'comprehensive_inference_results.npz')
        np.savez(results_path, **results)
        print(f"Comprehensive results saved to {results_path}")
        
    else:
        # For other datasets, use dataloader
        dataloaders = create_dataloaders(
            dataset_type=args.dataset,
            data_path=args.data_path,
            window_size=args.window_size,
            dataset_name=(args.dataset_name if hasattr(args, 'dataset_name') and args.dataset_name else config_dataset_name),
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
    reconstruction_errors, anomaly_scores, absolute_errors = inference.sliding_window_inference(
        test_data, args.window_size, args.stride, args.batch_size
    )
    
    # Create full reconstruction
    reconstruction = inference.create_full_reconstruction(
        test_data, reconstruction_errors, args.window_size, args.stride
    )
    
    # Compute per-timestep scores using new strategy:
    # Window 0: all timesteps, Window 1+: only last timestep
    timestep_scores = inference.compute_timestep_scores(
        absolute_errors, test_data.shape[1], args.window_size, args.stride
    )
    
    # Ensure labels and timestep_scores have the same length
    if labels is not None and len(labels) != len(timestep_scores):
        print(f"Info: Labels length ({len(labels)}) != Timestep scores length ({len(timestep_scores)})")
        print(f"This is normal for sliding window inference. Adjusting labels to match timestep scores length...")
        
        # Adjust labels to match timestep_scores length
        if len(labels) > len(timestep_scores):
            # Truncate labels to match timestep scores (common case)
            original_length = len(labels)
            labels = labels[:len(timestep_scores)]
            print(f"Truncated labels from {original_length} to {len(labels)}")
        else:
            # Pad labels with the last label value (rare case)
            original_length = len(labels)
            last_label = labels[-1] if len(labels) > 0 else 0
            padding = np.full(len(timestep_scores) - len(labels), last_label)
            labels = np.concatenate([labels, padding])
            print(f"Padded labels from {original_length} to {len(labels)}")
        
        print(f"Final labels length: {len(labels)}")
    
    # Evaluate performance across threshold range
    print(f"\nEvaluating performance across {args.num_thresholds} threshold values...")
    threshold_results = inference.evaluate_threshold_range(
        timestep_scores, labels, 
        num_thresholds=args.num_thresholds, use_adjustment=True
    )
    
    # Use best threshold for final results
    threshold_value = threshold_results['best_threshold']
    timestep_anomalies = timestep_scores > threshold_value
    
    # Convert timestep anomalies to window anomalies
    anomalies = np.zeros(len(anomaly_scores), dtype=bool)
    for i, is_anomaly in enumerate(timestep_anomalies):
        if is_anomaly:
            window_idx = i // args.stride
            if window_idx < len(anomalies):
                anomalies[window_idx] = True
                
    performance = threshold_results['best_metrics']
    
    # Plot threshold analysis
    threshold_plot_path = os.path.join(args.output_dir, 'threshold_analysis.png') if args.save_plot else None
    inference.plot_threshold_analysis(threshold_results, threshold_plot_path)
        
    # Save results to Excel
    if args.save_excel:
        # Use a fixed Excel filename to accumulate results from multiple models
        # Use distinct filename for single-file vs multi-file runs
        if hasattr(args, 'dataset_name') and args.dataset_name:
            base = os.path.splitext(os.path.basename(args.dataset_name))[0]
            excel_path = os.path.join(args.output_dir, f'{args.dataset}_single_{base}_results.xlsx')
        else:
            excel_path = os.path.join(args.output_dir, f'{args.dataset}_all_models_results.xlsx')
        
        # Load additional information from config and checkpoint
        additional_info = {}
        
        # Load config information
        config_path = os.path.join(os.path.dirname(args.model_path), 'config.json')
        if os.path.exists(config_path):
            try:
                import json
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    additional_info.update({
                    'Use_Contrastive': config_data.get('use_contrastive', 'Unknown'),
                    'Contrastive_Weight': config_data.get('contrastive_weight', 'Unknown'),
                    'Reconstruction_Weight': config_data.get('reconstruction_weight', 'Unknown'),
                    'Temperature': config_data.get('temperature', 'Unknown'),
                    'D_Model': config_data.get('d_model', 'Unknown'),
                    'Projection_Dim': config_data.get('projection_dim', 'Unknown'),
                    'Transformer_Layers': config_data.get('transformer_layers', 'Unknown'),
                    'TCN_Layers': config_data.get('tcn_num_layers', 'Unknown'),
                    'Learning_Rate': config_data.get('learning_rate', 'Unknown'),
                    'Weight_Decay': config_data.get('weight_decay', 'Unknown'),
                    'Epsilon': config_data.get('epsilon', 'Unknown'),
                    'Training_Batch_Size': config_data.get('batch_size', 'Unknown'),
                    'Num_Epochs': config_data.get('num_epochs', 'Unknown'),
                    'Training_Device': config_data.get('device', 'Unknown'),
                    'Seed': config_data.get('seed', 'Unknown'),
                    'Mask_Mode': config_data.get('mask_mode', 'Unknown'),
                    'Mask_Ratio': config_data.get('mask_ratio', 'Unknown'),
                    'Mask_Seed': config_data.get('mask_seed', 'Unknown')
                    })
            except Exception as e:
                print(f"Warning: Could not load config.json: {e}")
        
        # Load checkpoint information for best loss
        try:
            import torch
            checkpoint = torch.load(args.model_path, map_location='cpu')
            additional_info.update({
                'Best_Training_Loss': checkpoint.get('best_loss', 'Unknown'),
                'Training_Epoch': checkpoint.get('epoch', 'Unknown'),
                'Has_Optimizer': checkpoint.get('optimizer_state_dict') is not None,
                'Has_Scheduler': checkpoint.get('scheduler_state_dict') is not None,
                'Contrastive_Weight_Used': checkpoint.get('contrastive_weight', 'Unknown'),
                'Reconstruction_Weight_Used': checkpoint.get('reconstruction_weight', 'Unknown')
            })
        except Exception as e:
            print(f"Warning: Could not load checkpoint info: {e}")
        
        # Prepare config information
        config_info = {
            'Dataset_Type': args.dataset,
            'Data_Path': args.data_path,
            'Model_Path': args.model_path,
            'Window_Size': args.window_size,
            'Stride': args.stride,
            'Inference_Batch_Size': args.batch_size,
            'Num_Thresholds': args.num_thresholds,
            'Use_Threshold_Range': True,
            'Mask_Mode': args.mask_mode,
            'Mask_Ratio': args.mask_ratio,
            'Mask_Seed': args.mask_seed,
            'Save_Plot': args.save_plot,
            'Save_Excel': args.save_excel,
            'Output_Directory': args.output_dir,
            'Input_Dimension': inference.input_dim,
            'Model_Parameters': sum(p.numel() for p in inference.model.parameters()),
            'Device': inference.device,
            'Model_Architecture': 'ContrastiveModel',
            # Encoder hyperparameters
            'Enc_d_model': getattr(inference.model, 'd_model', 'Unknown'),
            'Enc_nhead': None,  # Not stored directly on model; best-effort via config above
            'Enc_transformer_layers': None,
            'Enc_tcn_kernel_size': None,
            'Enc_tcn_num_layers': None,
            # Augmentation hyperparameters (from config if present)
            'Aug_nhead': additional_info.get('Aug_nhead', 'Unknown'),
            'Aug_num_layers': additional_info.get('Aug_num_layers', 'Unknown'),
            'Aug_tcn_kernel_size': additional_info.get('Aug_tcn_kernel_size', 'Unknown'),
            'Aug_tcn_num_layers': additional_info.get('Aug_tcn_num_layers', 'Unknown'),
            'Aug_dropout': additional_info.get('Aug_dropout', 'Unknown'),
            'Aug_temperature': additional_info.get('Aug_temperature', 'Unknown'),
            'Best_Threshold': threshold_results['best_threshold'],
            'Best_F1_Score': threshold_results['best_f1'],
            'Best_Precision': threshold_results['best_metrics']['precision'],
            'Best_Recall': threshold_results['best_metrics']['recall'],
            'Best_Accuracy': threshold_results['best_metrics']['accuracy']
        }
        
        # Merge additional info
        config_info.update(additional_info)
        
        inference.save_threshold_results_to_excel(threshold_results, excel_path, config_info)
        
        # Create model comparison sheet if multiple models exist
        inference.create_model_comparison_sheet(excel_path)
        
        # Plot results with best threshold
        best_plot_path = os.path.join(args.output_dir, 'best_threshold_results.png') if args.save_plot else None
        # Check if best_metrics has the correct keys
        best_metrics = threshold_results['best_metrics']
        if 'f1' not in best_metrics:
            print(f"Warning: best_metrics missing 'f1' key. Available keys: {list(best_metrics.keys())}")
            # Try to fix if it has 'f1_score' instead
            if 'f1_score' in best_metrics:
                best_metrics['f1'] = best_metrics['f1_score']
                print("Fixed: copied 'f1_score' to 'f1'")
            else:
                best_metrics['f1'] = 0.0
                print("Added default 'f1' = 0.0")
        
        inference.plot_best_threshold_results(
            test_data, reconstruction, labels, timestep_scores,
            threshold_results['best_threshold'], best_metrics,
            best_plot_path
        )
        
    
    # Evaluate performance
    if labels is not None:
        performance = inference.evaluate_performance(
            labels, anomalies, args.window_size, args.stride, use_adjustment=True
        )
        print("\n" + "="*60)
        print("ECG ANOMALY DETECTION PERFORMANCE")
        print("="*60)
        print(f"🎯 F1-Score: {performance['f1_score']:.4f}")
        print(f"📊 Precision: {performance['precision']:.4f}")
        print(f"📈 Recall: {performance['recall']:.4f}")
        print(f"✅ Accuracy: {performance['accuracy']:.4f}")
        print("-" * 60)
        print(f"True Positives: {performance['true_positives']}")
        print(f"False Positives: {performance['false_positives']}")
        print(f"False Negatives: {performance['false_negatives']}")
        print(f"True Negatives: {performance['true_negatives']}")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("ECG ANOMALY DETECTION STATISTICS")
        print("="*60)
        print(f"Total windows processed: {len(anomaly_scores)}")
        print(f"Anomalous windows detected: {np.sum(anomalies)}")
        print(f"Anomaly rate: {np.sum(anomalies) / len(anomalies) * 100:.2f}%")
        print(f"Mean anomaly score: {np.mean(anomaly_scores):.6f}")
        print(f"Std anomaly score: {np.std(anomaly_scores):.6f}")
        print(f"Min anomaly score: {np.min(anomaly_scores):.6f}")
        print(f"Max anomaly score: {np.max(anomaly_scores):.6f}")
        print("="*60)
    
    # Plot results
    plot_path = os.path.join(args.output_dir, 'inference_results.png') if args.save_plot else None
    inference.plot_results(
        test_data, reconstruction, labels, anomaly_scores, 
        anomalies, args.window_size, args.stride, threshold_value, plot_path,
        timestep_scores=timestep_scores, timestep_anomalies=timestep_anomalies
    )
    
    # Save results
    results = {
        'reconstruction_errors': reconstruction_errors,
        'anomaly_scores': anomaly_scores,
        'absolute_errors': absolute_errors,
        'anomalies': anomalies,
        'reconstruction': reconstruction,
        'performance': performance if labels is not None else None,
        'threshold_value': float(threshold_value),
        'timestep_scores': timestep_scores,
        'timestep_anomalies': timestep_anomalies
    }
    
    # Add threshold results
    if 'threshold_results' in locals():
        results['threshold_results'] = threshold_results
    
    results_path = os.path.join(args.output_dir, 'inference_results.npz')
    np.savez(results_path, **results)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
