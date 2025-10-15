import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from typing import Dict, List, Tuple, Optional
from scipy.stats import zscore

# Import fixed dataset loaders
# Handle both direct execution and module execution
try:
    from .dataset_loaders_fixed import get_dataset_loader
except ImportError:
    # Fallback for direct execution - use absolute imports
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.utils.dataset_loaders_fixed import get_dataset_loader


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


