#!/usr/bin/env python3
"""
Fixed dataset loaders that properly extract labels from data
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict

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
            train_min = np.min(train_features)
            train_max = np.max(train_features)
            if train_max > train_min:
                train_features = (train_features - train_min) / (train_max - train_min)
                test_features = (test_features - train_min) / (train_max - train_min)
        
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
            train_min = np.min(train_features)
            train_max = np.max(train_features)
            if train_max > train_min:
                train_features = (train_features - train_min) / (train_max - train_min)
                test_features = (test_features - train_min) / (train_max - train_min)
        
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
            train_min = np.min(train_features)
            train_max = np.max(train_features)
            if train_max > train_min:
                train_features = (train_features - train_min) / (train_max - train_min)
                test_features = (test_features - train_min) / (train_max - train_min)
        
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
            train_min = np.min(train_data)
            train_max = np.max(train_data)
            if train_max > train_min:
                train_data = (train_data - train_min) / (train_max - train_min)
                test_data = (test_data - train_min) / (train_max - train_min)
        
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
            train_min = np.min(train_data)
            train_max = np.max(train_data)
            if train_max > train_min:
                train_data = (train_data - train_min) / (train_max - train_min)
                test_data = (test_data - train_min) / (train_max - train_min)
        
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
            train_min = np.min(train_data)
            train_max = np.max(train_data)
            if train_max > train_min:
                train_data = (train_data - train_min) / (train_max - train_min)
                test_data = (test_data - train_min) / (train_max - train_min)
        
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
            train_min = np.min(train_data)
            train_max = np.max(train_data)
            if train_max > train_min:
                train_data = (train_data - train_min) / (train_max - train_min)
                test_data = (test_data - train_min) / (train_max - train_min)
        
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
