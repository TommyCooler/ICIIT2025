import os
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from typing import Dict, List, Tuple, Optional, Union
from abc import ABC, abstractmethod
import json
import random
from scipy import signal
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# DataAugmentation class removed - use augmentation from src/modules/augmentation.py instead


class DataPreprocessing:
    """Data preprocessing techniques for time series data"""
    
    @staticmethod
    def z_score_normalize(data: np.ndarray) -> np.ndarray:
        """Z-score normalization"""
        normalized_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            normalized_data[i] = zscore(data[i], nan_policy='omit')
        return normalized_data

    @staticmethod
    def min_max_normalize(data: np.ndarray) -> np.ndarray:
        """Min-max normalization to [0, 1]"""
        normalized_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            data_min = np.min(data[i])
            data_max = np.max(data[i])
            if data_max > data_min:
                normalized_data[i] = (data[i] - data_min) / (data_max - data_min)
        return normalized_data
    
    @staticmethod
    def robust_normalize(data: np.ndarray) -> np.ndarray:
        """Robust normalization using median and IQR"""
        normalized_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            median = np.median(data[i])
            q75, q25 = np.percentile(data[i], [75, 25])
            iqr = q75 - q25
            if iqr > 0:
                normalized_data[i] = (data[i] - median) / iqr
        return normalized_data
    
    @staticmethod
    def detrend(data: np.ndarray) -> np.ndarray:
        """Remove linear trend from the data"""
        detrended_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            detrended_data[i] = signal.detrend(data[i])
        return detrended_data
    
    @staticmethod
    def smooth(data: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Apply moving average smoothing"""
        smoothed_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            smoothed_data[i] = np.convolve(data[i], np.ones(window_size)/window_size, mode='same')
        return smoothed_data


class BaseDataset(Dataset, ABC):
    """Base class for all dataset loaders"""
    
    def __init__(self, data: np.ndarray, labels: Optional[np.ndarray] = None, 
                 window_size: int = 100, stride: int = 1, normalize: bool = True,
                 preprocessing: Optional[str] = None):
        """
        Args:
            data: Input time series data of shape (features, time_steps)
            labels: Anomaly labels of shape (time_steps,)
            window_size: Size of sliding window
            stride: Step size for sliding window
            normalize: Whether to normalize the data
            preprocessing: Type of preprocessing ('zscore', 'minmax', 'robust', 'detrend', 'smooth')
        """
        self.data = data
        self.labels = labels
        self.window_size = window_size
        self.stride = stride
        self.normalize = normalize
        self.preprocessing = preprocessing
        
        # Apply preprocessing
        if self.preprocessing:
            self._apply_preprocessing()
        
        if self.normalize:
            self._normalize_data()
        
        # Calculate number of samples
        self.n_samples = (self.data.shape[1] - self.window_size) // self.stride + 1
        
    def _apply_preprocessing(self):
        """Apply specified preprocessing technique"""
        if self.preprocessing == 'zscore':
            self.data = DataPreprocessing.z_score_normalize(self.data)
        elif self.preprocessing == 'minmax':
            self.data = DataPreprocessing.min_max_normalize(self.data)
        elif self.preprocessing == 'robust':
            self.data = DataPreprocessing.robust_normalize(self.data)
        elif self.preprocessing == 'detrend':
            self.data = DataPreprocessing.detrend(self.data)
        elif self.preprocessing == 'smooth':
            self.data = DataPreprocessing.smooth(self.data)
        
    def _normalize_data(self):
        """Normalize data to [0, 1] range globally"""
        data_min = np.min(self.data)
        data_max = np.max(self.data)
        if data_max > data_min:
            self.data = (self.data - data_min) / (data_max - data_min)
    
# Augmentation is now handled by the contrastive model, not in dataloader
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size
        
        # Get window data
        window_data = self.data[:, start_idx:end_idx]  # Shape: (features, window_size)
        
        # Convert to tensor and transpose to (window_size, features)
        window_data = torch.FloatTensor(window_data.T)
        
        if self.labels is not None:
            # For labels, we use the label at the end of the window
            label = self.labels[end_idx - 1]
            return window_data, torch.LongTensor([label])
        else:
            return window_data


class ECGDatasetLoader:
    """Loader for ECG datasets (pkl files)"""
    
    def __init__(self, data_path: str, normalize: bool = True, validation_ratio: float = 0.2):
        self.data_path = data_path
        self.normalize = normalize
        self.validation_ratio = validation_ratio
        
    def load_dataset(self, dataset_name: str) -> Dict[str, Union[np.ndarray, BaseDataset]]:
        """Load a specific ECG dataset"""
        # Load training and test data
        train_path = os.path.normpath(os.path.join(self.data_path, "labeled", "train", dataset_name))
        test_path = os.path.normpath(os.path.join(self.data_path, "labeled", "test", dataset_name))
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Train file not found: {train_path}")
        
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test file not found: {test_path}")
        
        import pickle
        with open(train_path, 'rb') as f:
            train_data = pickle.load(f)
        with open(test_path, 'rb') as f:
            test_data = pickle.load(f)
        
        # Check if data is already in the correct format
        if isinstance(train_data, list):
            # Convert list to numpy array if needed
            train_data = np.array(train_data)
        if isinstance(test_data, list):
            test_data = np.array(test_data)
        
        # Extract signal data - automatically detect shape
        if train_data.shape[0] <= train_data.shape[1]:
            # Data is in [features, time] format
            pass
        else:
            # Data is in [time, features] format, transpose it
            train_data = train_data.T
            test_data = test_data.T
        
        # Extract features and labels from ECG data
        # ECG data format: [feature1, feature2, label] where label: 0=normal, 1=anomaly
        if train_data.shape[0] >= 3:  # Has at least 3 rows (2 features + 1 label)
            # Extract features (first 2 rows) and labels (third row)
            train_features = train_data[:2, :]  # First 2 rows are features
            train_labels = train_data[2, :]     # Third row is labels
            test_features = test_data[:2, :]    # First 2 rows are features
            test_labels = test_data[2, :]       # Third row is labels
            
            # Use only features for training (train set should be all normal)
            train_data = train_features
            test_data = test_features
            
            print(f"ECG dataset {dataset_name}:")
            print(f"  Train: {train_data.shape[0]} features, {train_data.shape[1]} time steps")
            print(f"  Train labels: {np.sum(train_labels == 0)} normal, {np.sum(train_labels == 1)} anomaly")
            print(f"  Test: {test_data.shape[0]} features, {test_data.shape[1]} time steps")
            print(f"  Test labels: {np.sum(test_labels == 0)} normal, {np.sum(test_labels == 1)} anomaly")
        else:
            # Fallback: create dummy labels if no labels available
            test_labels = np.zeros(test_data.shape[1])  # Dummy labels
            print(f"ECG dataset {dataset_name}: No labels found, using dummy labels")
        
        # Normalize if required
        if self.normalize:
            # Global normalization to [0, 1] using train data statistics
            train_min = np.min(train_data)
            train_max = np.max(train_data)
            if train_max > train_min:
                train_data = (train_data - train_min) / (train_max - train_min)
                test_data = (test_data - train_min) / (train_max - train_min)
        
        # Split test into validation and test sets
        val_size = int(test_data.shape[1] * self.validation_ratio)
        val_data = test_data[:, :val_size]
        val_labels = test_labels[:val_size]
        test_data = test_data[:, val_size:]
        test_labels = test_labels[val_size:]
        
        return {
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data,
            'val_labels': val_labels,
            'test_labels': test_labels
        }
    
    def load_all_datasets(self) -> Dict[str, Union[np.ndarray, BaseDataset]]:
        """Load all ECG datasets and combine them"""
        dataset_names = [
            'chfdb_chf01_275.pkl', 'chfdb_chf13_45590.pkl', 'chfdbchf15.pkl',
            'ltstdb_20221_43.pkl', 'ltstdb_20321_240.pkl', 'mitdb__100_180.pkl',
            'qtdbsel102.pkl', 'stdb_308_0.pkl', 'xmitdb_x108_0.pkl'
        ]
        
        all_datasets = []
        print(f"Data path: {self.data_path}")
        
        for dataset_name in dataset_names:
            train_path = os.path.join(self.data_path, "labeled", "train", dataset_name)
            test_path = os.path.join(self.data_path, "labeled", "test", dataset_name)
            
            # Normalize paths for consistent display
            train_path = os.path.normpath(train_path)
            test_path = os.path.normpath(test_path)
            
            print(f"Checking dataset: {dataset_name}")
            print(f"  Train path: {train_path}")
            print(f"  Test path: {test_path}")
            print(f"  Train path exists: {os.path.exists(train_path)}")
            print(f"  Test path exists: {os.path.exists(test_path)}")
            
            if os.path.exists(train_path) and os.path.exists(test_path):
                try:
                    ds = self.load_dataset(dataset_name)
                    all_datasets.append(ds)
                    print(f"  + Successfully loaded {dataset_name}")
                except Exception as e:
                    print(f"  - Failed to load {dataset_name}: {e}")
            else:
                print(f"  - Missing files for {dataset_name}")
        
        print(f"Total datasets loaded: {len(all_datasets)}")
        
        if len(all_datasets) == 0:
            raise ValueError("No datasets could be loaded. Please check the data paths and file existence.")
        
        # Combine all datasets
        train_data = np.concatenate([ds['train_data'] for ds in all_datasets], axis=1)
        val_data = np.concatenate([ds['val_data'] for ds in all_datasets], axis=1)
        test_data = np.concatenate([ds['test_data'] for ds in all_datasets], axis=1)
        val_labels = np.concatenate([ds['val_labels'] for ds in all_datasets], axis=0)
        test_labels = np.concatenate([ds['test_labels'] for ds in all_datasets], axis=0)
        
        print(f"Combined data shapes:")
        print(f"  Train: {train_data.shape}")
        print(f"  Val: {val_data.shape}")
        print(f"  Test: {test_data.shape}")
        
        return {
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data,
            'val_labels': val_labels,
            'test_labels': test_labels
        }


class PSMDatasetLoader:
    """Loader for PSM datasets (CSV files)"""
    
    def __init__(self, data_path: str, normalize: bool = True, validation_ratio: float = 0.2):
        self.data_path = data_path
        self.normalize = normalize
        self.validation_ratio = validation_ratio
        
    def load_dataset(self) -> Dict[str, Union[np.ndarray, BaseDataset]]:
        """Load PSM dataset"""
        print("Loading PSM dataset")
        
        # Load data
        train_path = os.path.normpath(os.path.join(self.data_path, "train.csv"))
        test_path = os.path.normpath(os.path.join(self.data_path, "test.csv"))
        test_labels_path = os.path.normpath(os.path.join(self.data_path, "test_label.csv"))
        
        print(f"  Train path: {train_path}")
        print(f"  Test path: {test_path}")
        print(f"  Test labels path: {test_labels_path}")
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Train file not found: {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test file not found: {test_path}")
        if not os.path.exists(test_labels_path):
            raise FileNotFoundError(f"Test labels file not found: {test_labels_path}")
        
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        test_labels_df = pd.read_csv(test_labels_path)
        
        # Extract features (exclude timestamp column)
        feature_cols = [col for col in train_df.columns if col.startswith('feature_')]
        
        train_data = train_df[feature_cols].to_numpy().T  # Shape: [features, time]
        test_data = test_df[feature_cols].to_numpy().T
        test_labels = test_labels_df['label'].to_numpy()
        
        # Handle NaN values before normalization
        print(f"  Handling NaN values in PSM data...")
        train_nan_count = np.isnan(train_data).sum()
        test_nan_count = np.isnan(test_data).sum()
        print(f"  Train NaN count before handling: {train_nan_count}")
        print(f"  Test NaN count before handling: {test_nan_count}")
        
        if train_nan_count > 0:
            # Replace NaN values with forward fill, then backward fill
            for i in range(train_data.shape[0]):  # For each feature
                feature_data = train_data[i]
                # Forward fill
                mask = ~np.isnan(feature_data)
                if np.any(mask):
                    train_data[i] = np.interp(
                        np.arange(len(feature_data)),
                        np.arange(len(feature_data))[mask],
                        feature_data[mask]
                    )
                else:
                    # If all values are NaN, fill with 0
                    train_data[i] = 0.0
        
        if test_nan_count > 0:
            # Replace NaN values with forward fill, then backward fill
            for i in range(test_data.shape[0]):  # For each feature
                feature_data = test_data[i]
                # Forward fill
                mask = ~np.isnan(feature_data)
                if np.any(mask):
                    test_data[i] = np.interp(
                        np.arange(len(feature_data)),
                        np.arange(len(feature_data))[mask],
                        feature_data[mask]
                    )
                else:
                    # If all values are NaN, fill with 0
                    test_data[i] = 0.0
        
        print(f"  Train NaN count after handling: {np.isnan(train_data).sum()}")
        print(f"  Test NaN count after handling: {np.isnan(test_data).sum()}")
        
        # Normalize if required
        if self.normalize:
            # Global normalization to [0, 1] using train data statistics
            train_min = np.min(train_data)
            train_max = np.max(train_data)
            if train_max > train_min:
                train_data = (train_data - train_min) / (train_max - train_min)
                test_data = (test_data - train_min) / (train_max - train_min)
        
        # Split test into validation and test sets
        val_size = int(test_data.shape[1] * self.validation_ratio)
        val_data = test_data[:, :val_size]
        val_labels = test_labels[:val_size]
        test_data = test_data[:, val_size:]
        test_labels = test_labels[val_size:]
        
        return {
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data,
            'val_labels': val_labels,
            'test_labels': test_labels
        }


class SMAPMSLDatasetLoader:
    """Loader for SMAP/MSL datasets (npy files)"""
    
    def __init__(self, data_path: str, normalize: bool = True, validation_ratio: float = 0.2):
        self.data_path = data_path
        self.normalize = normalize
        self.validation_ratio = validation_ratio
        
    def load_dataset(self, dataset_name: str) -> Dict[str, Union[np.ndarray, BaseDataset]]:
        """Load SMAP/MSL dataset from processed folder (expects *_train.npy, *_test.npy, *_labels.npy)."""
        print(f"Loading SMAP/MSL dataset (processed): {dataset_name}")

        # Enforce specific selection as requested
        if dataset_name != 'C-2':
            raise ValueError("Only 'C-2' is allowed for MSL in processed mode. Please set --dataset_name C-2")

        proc_dir = os.path.normpath(os.path.join(self.data_path, "processed"))
        train_path = os.path.join(proc_dir, f"{dataset_name}_train.npy")
        test_path = os.path.join(proc_dir, f"{dataset_name}_test.npy")
        labels_path = os.path.join(proc_dir, f"{dataset_name}_labels.npy")

        print(f"  Train path: {train_path}")
        print(f"  Test path: {test_path}")
        print(f"  Labels path: {labels_path}")

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Train file not found: {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test file not found: {test_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        # Load arrays
        train_data = np.load(train_path)
        test_data = np.load(test_path)
        test_labels = np.load(labels_path)

        # Ensure shapes (features, time)
        if train_data.ndim != 2 or test_data.ndim != 2:
            raise ValueError(f"Unexpected shapes: train {train_data.shape}, test {test_data.shape}")

        if train_data.shape[0] > train_data.shape[1]:
            # (T, F) -> (F, T)
            train_data = train_data.T
        if test_data.shape[0] > test_data.shape[1]:
            test_data = test_data.T

        # Align labels to test time length
        if test_labels.ndim != 1:
            test_labels = test_labels.reshape(-1)
        if len(test_labels) != test_data.shape[1]:
            # Try to truncate or pad to match
            min_len = min(len(test_labels), test_data.shape[1])
            test_labels = test_labels[:min_len]
            test_data = test_data[:, :min_len]

        # Build validation split from test
        val_size = int(test_data.shape[1] * self.validation_ratio)
        val_data = test_data[:, :val_size]
        val_labels = test_labels[:val_size]
        test_data = test_data[:, val_size:]
        test_labels = test_labels[val_size:]

        # Normalize if required using train statistics
        if self.normalize:
            train_min = np.min(train_data)
            train_max = np.max(train_data)
            if train_max > train_min:
                train_data = (train_data - train_min) / (train_max - train_min)
                if val_data.size:
                    val_data = (val_data - train_min) / (train_max - train_min)
                test_data = (test_data - train_min) / (train_max - train_min)

        return {
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data,
            'val_labels': val_labels,
            'test_labels': test_labels
        }


class UCRDatasetLoader:
    """Loader for UCR datasets (labeled time series)"""
    
    def __init__(self, data_path: str, normalize: bool = True, validation_ratio: float = 0.2):
        self.data_path = data_path
        self.normalize = normalize
        self.validation_ratio = validation_ratio
        
    def load_dataset(self, dataset_name: str) -> Dict[str, Union[np.ndarray, BaseDataset]]:
        """Load a specific UCR dataset"""
        print(f"Loading UCR dataset: {dataset_name}")
        
        # UCR datasets are typically in labeled format
        data_path = os.path.normpath(os.path.join(self.data_path, f"{dataset_name}.npy"))
        
        print(f"  Data path: {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
        
        # Load data
        data = np.load(data_path)
        
        # UCR labeled data format: (time, features) or (time,)
        if data.ndim == 1:
            # Single feature time series
            data = data.reshape(-1, 1).T  # Shape: (1, time)
        elif data.ndim == 2:
            # Multi-feature time series
            data = data.T  # Shape: (features, time)
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")
        
        print(f"  Data shape: {data.shape}")
        
        # Check for NaN values
        nan_count = np.isnan(data).sum()
        inf_count = np.isinf(data).sum()
        print(f"  NaN count: {nan_count}")
        print(f"  Inf count: {inf_count}")
        
        if nan_count > 0:
            print("  Handling NaN values...")
            # Replace NaN values with forward fill
            for i in range(data.shape[0]):
                feature_data = data[i]
                mask = ~np.isnan(feature_data)
                if np.any(mask):
                    data[i] = np.interp(
                        np.arange(len(feature_data)),
                        np.arange(len(feature_data))[mask],
                        feature_data[mask]
                    )
                else:
                    data[i] = 0.0
        
        # Normalize if required
        if self.normalize:
            # Global normalization to [0, 1] using data statistics
            data_min = np.min(data)
            data_max = np.max(data)
            if data_max > data_min:
                data = (data - data_min) / (data_max - data_min)
        
        # For UCR, we don't have separate train/test, so we split the data
        total_time = data.shape[1]
        train_size = int(total_time * 0.8)  # 80% for training
        
        train_data = data[:, :train_size]
        test_data = data[:, train_size:]
        
        # Split test into validation and test sets
        val_size = int(test_data.shape[1] * self.validation_ratio)
        val_data = test_data[:, :val_size]
        test_data = test_data[:, val_size:]
        
        # Create dummy labels for UCR (since it's unsupervised)
        train_labels = np.zeros(train_data.shape[1])
        val_labels = np.zeros(val_data.shape[1])
        test_labels = np.zeros(test_data.shape[1])
        
        return {
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data,
            'train_labels': train_labels,
            'val_labels': val_labels,
            'test_labels': test_labels
        }


class SMDDatasetLoader:
    """Loader for SMD datasets (npy files)"""
    
    def __init__(self, data_path: str, normalize: bool = True, validation_ratio: float = 0.2):
        self.data_path = data_path
        self.normalize = normalize
        self.validation_ratio = validation_ratio
        
    def load_dataset(self, dataset_name: str) -> Dict[str, Union[np.ndarray, BaseDataset]]:
        """Load a specific SMD dataset"""
        print(f"Loading SMD dataset: {dataset_name}")
        
        # Load data
        data_path = os.path.normpath(os.path.join(self.data_path, f"{dataset_name}.npy"))
        print(f"  Data path: {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
        
        data = np.load(data_path)
        print(f"  Data shape: {data.shape}")
        
        # Split into train/test (80/20 split)
        split_idx = int(data.shape[0] * 0.8)
        train_data = data[:split_idx].T  # Shape: (features, time)
        test_data = data[split_idx:].T
        
        print(f"  Train data shape: {train_data.shape}")
        print(f"  Test data shape: {test_data.shape}")
        
        # For SMD, we don't have explicit labels, so we'll create synthetic ones
        # or use unsupervised methods
        test_labels = np.zeros(test_data.shape[1])
        
        # Normalize if required
        if self.normalize:
            # Global normalization to [0, 1] using train data statistics
            train_min = np.min(train_data)
            train_max = np.max(train_data)
            if train_max > train_min:
                train_data = (train_data - train_min) / (train_max - train_min)
                test_data = (test_data - train_min) / (train_max - train_min)
        
        # Split test into validation and test sets
        val_size = int(test_data.shape[1] * self.validation_ratio)
        val_data = test_data[:, :val_size]
        val_labels = test_labels[:val_size]
        test_data = test_data[:, val_size:]
        test_labels = test_labels[val_size:]
        
        return {
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data,
            'val_labels': val_labels,
            'test_labels': test_labels
        }


class DatasetFactory:
    """Factory class to create appropriate dataset loaders"""
    
    @staticmethod
    def create_loader(dataset_type: str, data_path: str, **kwargs) -> Union[
        ECGDatasetLoader, PSMDatasetLoader, 
        SMAPMSLDatasetLoader, SMDDatasetLoader, UCRDatasetLoader
    ]:
        """Create appropriate dataset loader based on dataset type"""
        
        loaders = {
            'ecg': ECGDatasetLoader,
            'psm': PSMDatasetLoader,
            'smap_msl': SMAPMSLDatasetLoader,
            'smd': SMDDatasetLoader,
            'ucr': UCRDatasetLoader
        }
        
        if dataset_type not in loaders:
            raise ValueError(f"Unknown dataset type: {dataset_type}. "
                           f"Available types: {list(loaders.keys())}")
        
        return loaders[dataset_type](data_path, **kwargs)


def create_dataloaders(dataset_type: str, data_path: str, 
                      window_size: int = 100, stride: int = 1,
                      batch_size: int = 32, num_workers: int = 4,
                      normalize: bool = True, validation_ratio: float = 0.2,
                      preprocessing: Optional[str] = None,
                      dataset_name: Optional[str] = None,
                      **kwargs) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test dataloaders for a given dataset type
    
    Args:
        dataset_type: Type of dataset ('ecg', 'psm', 'nab', 'smap_msl', 'smd')
        data_path: Path to dataset directory
        window_size: Size of sliding window
        stride: Step size for sliding window
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        normalize: Whether to normalize data
        validation_ratio: Ratio of test data to use for validation
        preprocessing: Type of preprocessing ('zscore', 'minmax', 'robust', 'detrend', 'smooth')
        dataset_name: Specific dataset name (for nab, smap_msl, smd)
        **kwargs: Additional arguments for specific dataset loaders
    
    Returns:
        Dictionary containing train, val, and test dataloaders
    """
    
    # Create dataset loader
    loader_kwargs = kwargs.copy()
    # Do NOT pass dataset_name to loader __init__; it's only used at load time
    loader = DatasetFactory.create_loader(
        dataset_type=dataset_type,
        data_path=data_path,
        normalize=normalize,
        validation_ratio=validation_ratio,
        **loader_kwargs
    )
    
    # Load dataset - enforce single-file mode (no cross-file concatenation)
    if dataset_type == 'psm':
        # PSM has fixed train/test files
        data = loader.load_dataset()
    else:
        if not dataset_name:
            raise ValueError(
                f"Single-file mode enforced: please provide --dataset_name for dataset type '{dataset_type}'."
            )
        data = loader.load_dataset(dataset_name)
    
    # Create datasets
    train_dataset = BaseDataset(
        data=data['train_data'],
        labels=None,  # No labels for training (unsupervised)
        window_size=window_size,
        stride=stride,
        normalize=False,  # Already normalized by loader
        preprocessing=preprocessing
    )
    
    val_dataset = BaseDataset(
        data=data['val_data'],
        labels=data['val_labels'],
        window_size=window_size,
        stride=stride,
        normalize=False,
        preprocessing=preprocessing
    )
    
    test_dataset = BaseDataset(
        data=data['test_data'],
        labels=data['test_labels'],
        window_size=window_size,
        stride=stride,
        normalize=False,
        preprocessing=preprocessing
    )
    
    # Create dataloaders
    dataloaders = {
        'train': DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True
        ),
        'val': DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        ),
        'test': DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
    }
    
    return dataloaders


# Example usage
if __name__ == "__main__":
    # Example for ECG dataset with preprocessing
    ecg_dataloaders = create_dataloaders(
        dataset_type='ecg',
        data_path='MainModel\\datasets\\ecg',
        window_size=100,
        stride=1,
        batch_size=32,
        preprocessing='zscore'
    )
    
    # Example for PSM dataset with different settings
    psm_dataloaders = create_dataloaders(
        dataset_type='psm',
        data_path='datasets\\psm',
        window_size=100,
        stride=1,
        batch_size=32,
        preprocessing='minmax'
    )
    
    # # Example for NAB dataset
    # nab_dataloaders = create_dataloaders(
    #     dataset_type='nab',
    #     data_path='datasets\\nab',
    #     dataset_name='ambient_temperature_system_failure',
    #     window_size=100,
    #     stride=1,
    #     batch_size=32,
    #     preprocessing='robust'
    # )
    
    print("Dataloaders created successfully!")
    print(f"ECG Train batches: {len(ecg_dataloaders['train'])}")
    print(f"ECG Val batches: {len(ecg_dataloaders['val'])}")
    print(f"ECG Test batches: {len(ecg_dataloaders['test'])}")
    
    # Test data loading
    for batch_idx, batch in enumerate(ecg_dataloaders['train']):
        if batch_idx == 0:
            print(f"\nSample batch shape: {batch.shape}")
            break
