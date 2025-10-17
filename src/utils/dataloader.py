#!/usr/bin/env python3
"""
Unified dataloader module combining dataset loaders and data processing utilities
"""

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from typing import Dict, List, Tuple, Optional
from scipy.stats import zscore
import sys
import os
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# =============================================================================
# DATASET LOADERS
# =============================================================================

class ECGDatasetLoader:
    """Loader for ECG datasets - labels are in the last column of data"""
    
    def __init__(self, data_path: str, normalize: bool = True):
        self.data_path = data_path
        self.normalize = normalize
        
    def load_dataset(self, dataset_name: str) -> Dict[str, np.ndarray]:
        """Load ECG dataset with labels extracted from data"""
        print(f"Loading ECG dataset: {dataset_name}")
        
        train_path = os.path.join(self.data_path, "labeled", "train", f"{dataset_name}.pkl")
        test_path = os.path.join(self.data_path, "labeled", "test", f"{dataset_name}.pkl")
        
        print(f"  Train path: {train_path}")
        print(f"  Test path: {test_path}")
        
        # Load data
        with open(train_path, 'rb') as f:
            train_data = pickle.load(f)  # List of lists
        
        with open(test_path, 'rb') as f:
            test_data = pickle.load(f)  # List of lists
        
        print(f"  Train data: {type(train_data)} - Length: {len(train_data)}")
        print(f"  Test data: {type(test_data)} - Length: {len(test_data)}")
        
        # Convert to numpy arrays
        train_array = np.array(train_data)  # (time_steps, 3) - [feature1, feature2, label]
        test_array = np.array(test_data)    # (time_steps, 3) - [feature1, feature2, label]
        
        print(f"  Train array shape: {train_array.shape}")
        print(f"  Test array shape: {test_array.shape}")
        
        # Extract features and labels
        train_features = train_array[:, :-1].T  # (2, time_steps)
        train_labels = train_array[:, -1]       # (time_steps,)
        
        test_features = test_array[:, :-1].T    # (2, time_steps)
        test_labels = test_array[:, -1]         # (time_steps,)
        
        print(f"  Train features shape: {train_features.shape}")
        print(f"  Train labels shape: {train_labels.shape}")
        print(f"  Train label values: {np.unique(train_labels)}")
        print(f"  Test features shape: {test_features.shape}")
        print(f"  Test labels shape: {test_labels.shape}")
        print(f"  Test label values: {np.unique(test_labels)}")
        
        # Normalize if required (only features, not labels)
        if self.normalize:
            # Use StandardScaler: fit on train, transform both train and test
            scaler = StandardScaler()
            # Fit scaler on train data (features only)
            train_features_scaled = scaler.fit_transform(train_features.T)  # (time_steps, features)
            # Transform test data using train statistics
            test_features_scaled = scaler.transform(test_features.T)  # (time_steps, features)
            # Transpose back to (features, time_steps)
            train_features = train_features_scaled.T
            test_features = test_features_scaled.T
        
        return {
            'train_data': train_features,
            'test_data': test_features,
            'train_labels': train_labels,
            'test_labels': test_labels
        }

class PDDatasetLoader:
    """Loader for PD datasets - labels are in the last column of data"""
    
    def __init__(self, data_path: str, normalize: bool = True):
        self.data_path = data_path
        self.normalize = normalize
        
    def load_dataset(self, dataset_name: str) -> Dict[str, np.ndarray]:
        """Load PD dataset with labels extracted from data"""
        print(f"Loading PD dataset: {dataset_name}")
        
        train_path = os.path.join(self.data_path, "labeled", "train", f"{dataset_name}.pkl")
        test_path = os.path.join(self.data_path, "labeled", "test", f"{dataset_name}.pkl")
        
        print(f"  Train path: {train_path}")
        print(f"  Test path: {test_path}")
        
        # Load data
        with open(train_path, 'rb') as f:
            train_data = pickle.load(f)  # List of lists
        
        with open(test_path, 'rb') as f:
            test_data = pickle.load(f)  # List of lists
        
        print(f"  Train data: {type(train_data)} - Length: {len(train_data)}")
        print(f"  Test data: {type(test_data)} - Length: {len(test_data)}")
        
        # Convert to numpy arrays
        train_array = np.array(train_data)  # (time_steps, 2) - [feature, label]
        test_array = np.array(test_data)    # (time_steps, 2) - [feature, label]
        
        print(f"  Train array shape: {train_array.shape}")
        print(f"  Test array shape: {test_array.shape}")
        
        # Extract features and labels
        train_features = train_array[:, :-1].T  # (1, time_steps)
        train_labels = train_array[:, -1]       # (time_steps,)
        
        test_features = test_array[:, :-1].T    # (1, time_steps)
        test_labels = test_array[:, -1]         # (time_steps,)
        
        print(f"  Train features shape: {train_features.shape}")
        print(f"  Train labels shape: {train_labels.shape}")
        print(f"  Train label values: {np.unique(train_labels)}")
        print(f"  Test features shape: {test_features.shape}")
        print(f"  Test labels shape: {test_labels.shape}")
        print(f"  Test label values: {np.unique(test_labels)}")
        
        # Normalize if required (only features, not labels)
        if self.normalize:
            # Use StandardScaler: fit on train, transform both train and test
            scaler = StandardScaler()
            # Fit scaler on train data (features only)
            train_features_scaled = scaler.fit_transform(train_features.T)  # (time_steps, features)
            # Transform test data using train statistics
            test_features_scaled = scaler.transform(test_features.T)  # (time_steps, features)
            # Transpose back to (features, time_steps)
            train_features = train_features_scaled.T
            test_features = test_features_scaled.T
        
        return {
            'train_data': train_features,
            'test_data': test_features,
            'train_labels': train_labels,
            'test_labels': test_labels
        }

class GestureDatasetLoader:
    """Loader for Gesture datasets - labels are in the last column of data"""
    
    def __init__(self, data_path: str, normalize: bool = True):
        self.data_path = data_path
        self.normalize = normalize
        
    def load_dataset(self, dataset_name: str) -> Dict[str, np.ndarray]:
        """Load Gesture dataset with labels extracted from data"""
        print(f"Loading Gesture dataset: {dataset_name}")
        
        train_path = os.path.join(self.data_path, "labeled", "train", f"{dataset_name}.pkl")
        test_path = os.path.join(self.data_path, "labeled", "test", f"{dataset_name}.pkl")
        
        print(f"  Train path: {train_path}")
        print(f"  Test path: {test_path}")
        
        # Load data
        with open(train_path, 'rb') as f:
            train_data = pickle.load(f)  # List of lists
        
        with open(test_path, 'rb') as f:
            test_data = pickle.load(f)  # List of lists
        
        print(f"  Train data: {type(train_data)} - Length: {len(train_data)}")
        print(f"  Test data: {type(test_data)} - Length: {len(test_data)}")
        
        # Convert to numpy arrays
        train_array = np.array(train_data)  # (time_steps, 3) - [feature1, feature2, label]
        test_array = np.array(test_data)    # (time_steps, 3) - [feature1, feature2, label]
        
        print(f"  Train array shape: {train_array.shape}")
        print(f"  Test array shape: {test_array.shape}")
        
        # Extract features and labels
        train_features = train_array[:, :-1].T  # (2, time_steps)
        train_labels = train_array[:, -1]       # (time_steps,)
        
        test_features = test_array[:, :-1].T    # (2, time_steps)
        test_labels = test_array[:, -1]         # (time_steps,)
        
        print(f"  Train features shape: {train_features.shape}")
        print(f"  Train labels shape: {train_labels.shape}")
        print(f"  Train label values: {np.unique(train_labels)}")
        print(f"  Test features shape: {test_features.shape}")
        print(f"  Test labels shape: {test_labels.shape}")
        print(f"  Test label values: {np.unique(test_labels)}")
        
        # Normalize if required (only features, not labels)
        if self.normalize:
            # Use StandardScaler: fit on train, transform both train and test
            scaler = StandardScaler()
            # Fit scaler on train data (features only)
            train_features_scaled = scaler.fit_transform(train_features.T)  # (time_steps, features)
            # Transform test data using train statistics
            test_features_scaled = scaler.transform(test_features.T)  # (time_steps, features)
            # Transpose back to (features, time_steps)
            train_features = train_features_scaled.T
            test_features = test_features_scaled.T
        
        return {
            'train_data': train_features,
            'test_data': test_features,
            'train_labels': train_labels,
            'test_labels': test_labels
        }

class PSMDatasetLoader:
    """Loader for PSM datasets - labels in separate CSV file"""
    
    def __init__(self, data_path: str, normalize: bool = True):
        self.data_path = data_path
        self.normalize = normalize
        
    def load_dataset(self, dataset_name: str = "psm") -> Dict[str, np.ndarray]:
        """Load PSM dataset with labels from separate file"""
        print(f"Loading PSM dataset")
        
        train_path = os.path.join(self.data_path, "train.csv")
        test_path = os.path.join(self.data_path, "test.csv")
        test_labels_path = os.path.join(self.data_path, "test_label.csv")
        
        print(f"  Train path: {train_path}")
        print(f"  Test path: {test_path}")
        print(f"  Test labels path: {test_labels_path}")
        
        # Load data
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        test_labels_df = pd.read_csv(test_labels_path)
        
        print(f"  Train DataFrame: {train_df.shape}")
        print(f"  Test DataFrame: {test_df.shape}")
        print(f"  Test Labels DataFrame: {test_labels_df.shape}")
        
        # Extract features (exclude timestamp column)
        feature_cols = [col for col in train_df.columns if col.startswith('feature_')]
        
        train_data = train_df[feature_cols].values.T  # (features, time_steps)
        test_data = test_df[feature_cols].values.T    # (features, time_steps)
        test_labels = test_labels_df['label'].values  # (time_steps,)
        
        print(f"  Feature columns: {len(feature_cols)}")
        print(f"  Final train shape: {train_data.shape}")
        print(f"  Final test shape: {test_data.shape}")
        print(f"  Test labels shape: {test_labels.shape}")
        print(f"  Test label values: {np.unique(test_labels)}")
        print(f"  Test label counts: {np.bincount(test_labels.astype(int))}")
        
        # Normalize if required
        if self.normalize:
            # Use StandardScaler: fit on train, transform both train and test
            scaler = StandardScaler()
            # Fit scaler on train data (features only)
            train_data_scaled = scaler.fit_transform(train_data.T)  # (time_steps, features)
            # Transform test data using train statistics
            test_data_scaled = scaler.transform(test_data.T)  # (time_steps, features)
            # Transpose back to (features, time_steps)
            train_data = train_data_scaled.T
            test_data = test_data_scaled.T
        
        # PSM datasets: train is normal (no labels), test has ground truth labels
        train_labels = np.zeros(train_data.shape[1])
        
        return {
            'train_data': train_data,
            'test_data': test_data,
            'train_labels': train_labels,
            'test_labels': test_labels
        }

class UCRDatasetLoader:
    """Loader for UCR datasets - numpy arrays format with separate label files"""
    
    def __init__(self, data_path: str, normalize: bool = True):
        self.data_path = data_path
        self.normalize = normalize
        
    def load_dataset(self, dataset_name: str) -> Dict[str, np.ndarray]:
        """Load UCR dataset"""
        print(f"Loading UCR dataset: {dataset_name}")
        
        labels_path = os.path.join(self.data_path, f"{dataset_name}_labels.npy")
        train_path = os.path.join(self.data_path, f"{dataset_name}_train.npy")
        test_path = os.path.join(self.data_path, f"{dataset_name}_test.npy")
        
        print(f"  Labels path: {labels_path}")
        print(f"  Train path: {train_path}")
        print(f"  Test path: {test_path}")
        
        # Load data
        train_data = np.load(train_path)  # (time_steps, features)
        test_data = np.load(test_path)    # (time_steps, features)
        labels = np.load(labels_path)     # (time_steps,)
        
        print(f"  Train data: {type(train_data)} - {train_data.shape}")
        print(f"  Test data: {type(test_data)} - {test_data.shape}")
        print(f"  Labels: {type(labels)} - {labels.shape}")
        
        # Transpose to (features, time_steps) format
        train_data = train_data.T
        test_data = test_data.T
        
        print(f"  Final train shape: {train_data.shape}")
        print(f"  Final test shape: {test_data.shape}")
        
        # Normalize if required
        if self.normalize:
            # Use StandardScaler: fit on train, transform both train and test
            scaler = StandardScaler()
            # Fit scaler on train data (features only)
            train_data_scaled = scaler.fit_transform(train_data.T)  # (time_steps, features)
            # Transform test data using train statistics
            test_data_scaled = scaler.transform(test_data.T)  # (time_steps, features)
            # Transpose back to (features, time_steps)
            train_data = train_data_scaled.T
            test_data = test_data_scaled.T
        
        # UCR datasets: train is normal (no labels), test has ground truth labels
        train_labels = np.zeros(train_data.shape[1])
        test_labels = labels.flatten()  # Ground truth labels for evaluation
        
        print(f"  Test label values: {np.unique(test_labels)}")
        print(f"  Test label counts: {np.bincount(test_labels.astype(int))}")
        
        return {
            'train_data': train_data,
            'test_data': test_data,
            'train_labels': train_labels,
            'test_labels': test_labels
        }

class NABDatasetLoader:
    """Loader for NAB datasets - numpy arrays format with separate label files"""
    
    def __init__(self, data_path: str, normalize: bool = True):
        self.data_path = data_path
        self.normalize = normalize
        
    def load_dataset(self, dataset_name: str) -> Dict[str, np.ndarray]:
        """Load NAB dataset"""
        print(f"Loading NAB dataset: {dataset_name}")
        
        train_path = os.path.join(self.data_path, f"{dataset_name}_train.npy")
        test_path = os.path.join(self.data_path, f"{dataset_name}_test.npy")
        labels_path = os.path.join(self.data_path, f"{dataset_name}_labels.npy")
        
        print(f"  Train path: {train_path}")
        print(f"  Test path: {test_path}")
        print(f"  Labels path: {labels_path}")
        
        # Load data
        train_data = np.load(train_path)  # (time_steps, features)
        test_data = np.load(test_path)    # (time_steps, features)
        labels = np.load(labels_path)     # (time_steps, 1)
        
        print(f"  Train data: {type(train_data)} - {train_data.shape}")
        print(f"  Test data: {type(test_data)} - {test_data.shape}")
        print(f"  Labels: {type(labels)} - {labels.shape}")
        
        # Transpose to (features, time_steps) format
        train_data = train_data.T
        test_data = test_data.T
        
        # Handle labels shape - flatten if needed
        if labels.ndim > 1:
            labels = labels.flatten()
        
        print(f"  Final train shape: {train_data.shape}")
        print(f"  Final test shape: {test_data.shape}")
        print(f"  Final labels shape: {labels.shape}")
        
        # Normalize if required
        if self.normalize:
            # Use StandardScaler: fit on train, transform both train and test
            scaler = StandardScaler()
            # Fit scaler on train data (features only)
            train_data_scaled = scaler.fit_transform(train_data.T)  # (time_steps, features)
            # Transform test data using train statistics
            test_data_scaled = scaler.transform(test_data.T)  # (time_steps, features)
            # Transpose back to (features, time_steps)
            train_data = train_data_scaled.T
            test_data = test_data_scaled.T
        
        # NAB datasets: train is normal (no labels), test has ground truth labels
        train_labels = np.zeros(train_data.shape[1])
        test_labels = labels  # Ground truth labels for evaluation
        
        print(f"  Test label values: {np.unique(test_labels)}")
        print(f"  Test label counts: {np.bincount(test_labels.astype(int))}")
        
        return {
            'train_data': train_data,
            'test_data': test_data,
            'train_labels': train_labels,
            'test_labels': test_labels
        }

class SMDDatasetLoader:
    """Loader for SMD datasets - numpy arrays format with separate label files"""
    
    def __init__(self, data_path: str, normalize: bool = True):
        self.data_path = data_path
        self.normalize = normalize
        
    def load_dataset(self, dataset_name: str) -> Dict[str, np.ndarray]:
        """Load SMD dataset"""
        print(f"Loading SMD dataset: {dataset_name}")
        
        train_path = os.path.join(self.data_path, f"{dataset_name}_train.npy")
        test_path = os.path.join(self.data_path, f"{dataset_name}_test.npy")
        labels_path = os.path.join(self.data_path, f"{dataset_name}_labels.npy")
        
        print(f"  Train path: {train_path}")
        print(f"  Test path: {test_path}")
        print(f"  Labels path: {labels_path}")
        
        # Load data
        train_data = np.load(train_path)  # (time_steps, features)
        test_data = np.load(test_path)    # (time_steps, features)
        labels = np.load(labels_path)     # (time_steps, features) - labels for each feature
        
        print(f"  Train data: {type(train_data)} - {train_data.shape}")
        print(f"  Test data: {type(test_data)} - {test_data.shape}")
        print(f"  Labels: {type(labels)} - {labels.shape}")
        
        # Transpose to (features, time_steps) format
        train_data = train_data.T
        test_data = test_data.T
        
        # For SMD, labels are per feature, so we need to aggregate them
        # Take the maximum label across features for each timestep (if any feature is anomalous, the timestep is anomalous)
        if labels.ndim > 1:
            labels = np.max(labels, axis=1)  # (time_steps,)
        
        print(f"  Final train shape: {train_data.shape}")
        print(f"  Final test shape: {test_data.shape}")
        print(f"  Final labels shape: {labels.shape}")
        
        # Normalize if required
        if self.normalize:
            # Use StandardScaler: fit on train, transform both train and test
            scaler = StandardScaler()
            # Fit scaler on train data (features only)
            train_data_scaled = scaler.fit_transform(train_data.T)  # (time_steps, features)
            # Transform test data using train statistics
            test_data_scaled = scaler.transform(test_data.T)  # (time_steps, features)
            # Transpose back to (features, time_steps)
            train_data = train_data_scaled.T
            test_data = test_data_scaled.T
        
        # SMD datasets: train is normal (no labels), test has ground truth labels
        train_labels = np.zeros(train_data.shape[1])
        test_labels = labels  # Ground truth labels for evaluation
        
        print(f"  Test label values: {np.unique(test_labels)}")
        print(f"  Test label counts: {np.bincount(test_labels.astype(int))}")
        
        return {
            'train_data': train_data,
            'test_data': test_data,
            'train_labels': train_labels,
            'test_labels': test_labels
        }

def get_dataset_loader(dataset_type: str, data_path: str, normalize: bool = True):
    """Factory function to get appropriate dataset loader"""
    
    loaders = {
        'ecg': ECGDatasetLoader,
        'pd': PDDatasetLoader,
        'gesture': GestureDatasetLoader,
        'ucr': UCRDatasetLoader,
        'psm': PSMDatasetLoader,
        'nab': NABDatasetLoader,
        'smd': SMDDatasetLoader
    }
    
    if dataset_type not in loaders:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Available: {list(loaders.keys())}")
    
    return loaders[dataset_type](data_path, normalize)


# =============================================================================
# DATA PREPROCESSING
# =============================================================================

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


# =============================================================================
# DATASET CLASSES
# =============================================================================

class BaseDataset(Dataset):
    """Base dataset class for time series data"""
    
    def __init__(self, 
                 data: np.ndarray, 
                 labels: Optional[np.ndarray] = None,
                 window_size: int = 100, 
                 stride: int = 1,
                 preprocessing: Optional[str] = None):
        """
        Args:
            data: Time series data of shape (features, time_steps)
            labels: Optional labels of shape (time_steps,)
            window_size: Size of sliding window
            stride: Step size for sliding window  
            preprocessing: Type of preprocessing to apply
        """
        self.data = data
        self.labels = labels
        self.window_size = window_size
        self.stride = stride
        
        # Apply preprocessing if specified
        if preprocessing:
            if preprocessing == 'zscore':
                self.data = DataPreprocessing.z_score_normalize(self.data)
            elif preprocessing == 'minmax':
                self.data = DataPreprocessing.min_max_normalize(self.data)
            elif preprocessing == 'robust':
                self.data = DataPreprocessing.robust_normalize(self.data)
        
        # Generate windows
        self.windows = self._generate_windows()
        
    def _generate_windows(self) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """Generate sliding windows from time series data"""
        windows = []
        time_steps = self.data.shape[1]
        
        for start_idx in range(0, time_steps - self.window_size + 1, self.stride):
            end_idx = start_idx + self.window_size
            
            # Extract window data
            window_data = self.data[:, start_idx:end_idx]  # (features, window_size)
            
            # Extract window labels if available
            window_labels = None
            if self.labels is not None:
                window_labels = self.labels[start_idx:end_idx]  # (window_size,)
            
            windows.append((window_data, window_labels))
        
        return windows
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Get window at index"""
        window_data, window_labels = self.windows[idx]
        
        # Convert to tensors
        window_tensor = torch.FloatTensor(window_data)
        labels_tensor = None
        if window_labels is not None:
            labels_tensor = torch.LongTensor(window_labels)
        
        return window_tensor, labels_tensor


class DatasetFactory:
    """Factory class to create appropriate dataset loaders"""
    
    @staticmethod
    def create_loader(dataset_type: str, data_path: str, **kwargs):
        """Create appropriate dataset loader based on dataset type"""
        return get_dataset_loader(dataset_type, data_path, **kwargs)


# =============================================================================
# MAIN DATALOADER FUNCTIONS
# =============================================================================

def create_dataloaders(dataset_type: str, data_path: str, 
                      window_size: int = 100, stride: int = 1,
                      batch_size: int = 32, num_workers: int = 4,
                      normalize: bool = True,
                      preprocessing: Optional[str] = None,
                      dataset_name: Optional[str] = None,
                      **kwargs) -> Dict[str, DataLoader]:
    """
    Create train and test dataloaders for a given dataset type
    
    Args:
        dataset_type: Type of dataset ('ecg', 'psm', 'nab', 'smap_msl', 'smd')
        data_path: Path to dataset directory
        window_size: Size of sliding window
        stride: Step size for sliding window
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        normalize: Whether to normalize data
        preprocessing: Type of preprocessing ('zscore', 'minmax', 'robust')
        dataset_name: Specific dataset name (for nab, smap_msl, smd)
        **kwargs: Additional arguments for specific dataset loaders
    
    Returns:
        Dictionary containing train and test dataloaders
    """
    
    # Create dataset loader
    loader_kwargs = kwargs.copy()
    loader = DatasetFactory.create_loader(
        dataset_type=dataset_type,
        data_path=data_path,
        normalize=normalize,
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
            preprocessing=preprocessing
        )
    
    test_dataset = BaseDataset(
        data=data['test_data'],
        labels=data['test_labels'],
        window_size=window_size,
        stride=stride,
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
        'test': DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
    }
    
    return dataloaders


def create_contrastive_dataloaders(dataset_type: str, data_path: str, 
                                 window_size: int = 100, stride: int = 1,
                                 batch_size: int = 32, num_workers: int = 4,
                                 normalize: bool = True,
                                 preprocessing: Optional[str] = None,
                                 dataset_name: Optional[str] = None,
                                 **kwargs) -> Dict[str, DataLoader]:
    """
    Create contrastive learning dataloaders with augmented data
    
    Args:
        dataset_type: Type of dataset
        data_path: Path to dataset directory
        window_size: Size of sliding window
        stride: Step size for sliding window
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        normalize: Whether to normalize data
        preprocessing: Type of preprocessing
        dataset_name: Specific dataset name
        **kwargs: Additional arguments
    
    Returns:
        Dictionary containing train and test dataloaders
    """
    # Use the same function as create_dataloaders for now
    # Augmentation will be handled in the model forward pass
    return create_dataloaders(
        dataset_type=dataset_type,
        data_path=data_path,
        window_size=window_size,
        stride=stride,
        batch_size=batch_size,
        num_workers=num_workers,
        normalize=normalize,
        preprocessing=preprocessing,
        dataset_name=dataset_name,
        **kwargs
    )