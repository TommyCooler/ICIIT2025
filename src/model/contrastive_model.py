import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional
from modules.encoder import Encoder
from modules.decoder import Decoder
from modules.augmentation import Augmentation


class MLPProjection(nn.Module):
    """MLP for contrastive learning projection"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super(MLPProjection, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Output tensor of shape (batch_size, seq_len, output_dim)
        """
        return self.mlp(x)


class ContrastiveModel(nn.Module):
    """Contrastive Learning Model with TCN + Transformer Encoder"""
    
    def __init__(self,
                 input_dim: int,
                 d_model: int = 256,
                 projection_dim: int = 128,
                 nhead: int = 8,
                 transformer_layers: int = 6,
                 tcn_output_dim: Optional[int] = None,
                 tcn_kernel_size: int = 3,
                 tcn_num_layers: int = 3,
                 dropout: float = 0.1,
                 temperature: float = 1,
                 combination_method: str = 'concat',
                 use_contrastive: bool = True,
                 augmentation_kwargs: Optional[Dict] = None):
        """
        Args:
            input_dim: Input dimension (number of features)
            d_model: Model dimension for transformer
            projection_dim: Dimension for contrastive learning projection
            nhead: Number of attention heads
            transformer_layers: Number of transformer encoder layers
            tcn_output_dim: Output dimension for TCN
            tcn_kernel_size: Kernel size for TCN
            tcn_num_layers: Number of TCN layers
            dropout: Dropout rate
            temperature: Temperature for InfoNCE loss
            combination_method: 'concat' or 'stack' for encoder combination
            use_contrastive: Whether to use contrastive learning branch
        """
        super(ContrastiveModel, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.use_contrastive = use_contrastive
        
        # Encoder for both original and augmented data
        self.encoder = Encoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            transformer_layers=transformer_layers,
            tcn_output_dim=tcn_output_dim,
            tcn_kernel_size=tcn_kernel_size,
            tcn_num_layers=tcn_num_layers,
            dropout=dropout,
            combination_method=combination_method
        )
        
        # MLP for contrastive learning projection
        # Projection head for contrastive learning (only if using contrastive)
        if self.use_contrastive:
            self.projection_mlp = MLPProjection(
                input_dim=d_model,
                hidden_dim=d_model,
                output_dim=projection_dim,
                dropout=dropout
            )
        
        # MLP Decoder for reconstruction
        self.decoder = Decoder(
            d_model=d_model,
            output_dim=input_dim,
            hidden_dims=[d_model, d_model // 2, d_model // 4],
            dropout=dropout
        )
        
        # Augmentation module for data augmentation
        # Allow augmentation-specific overrides via augmentation_kwargs while keeping backward compatibility
        aug_kwargs = augmentation_kwargs.copy() if augmentation_kwargs is not None else {}
        aug_dropout = aug_kwargs.pop('dropout', dropout)
        aug_temperature = aug_kwargs.pop('temperature', temperature)
        # Provide sensible defaults from model config if not explicitly provided
        if 'nhead' not in aug_kwargs:
            aug_kwargs['nhead'] = nhead
        if 'tcn_kernel_size' not in aug_kwargs:
            aug_kwargs['tcn_kernel_size'] = tcn_kernel_size
        if 'num_layers' not in aug_kwargs:
            # Default augmentation transformer num layers = 1 unless overridden
            aug_kwargs['num_layers'] = 1
        
        self.augmentation = Augmentation(
            input_dim=input_dim,
            output_dim=input_dim,
            dropout=aug_dropout,
            temperature=aug_temperature,
            **aug_kwargs
        )
        
    def forward(self, original_data: torch.Tensor, augmented_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model
        
        Args:
            original_data: Original data tensor of shape (batch_size, seq_len, input_dim)
            augmented_data: Augmented data tensor of shape (batch_size, seq_len, input_dim)
        
        Returns:
            Dictionary containing:
                - original_encoded: Encoded original data
                - augmented_encoded: Encoded augmented data
                - original_projection: Projected original data for contrastive learning
                - augmented_projection: Projected augmented data for contrastive learning
                - reconstructed: Reconstructed original data from augmented encoding
        """
        # Apply augmentation to the data
        augmented_data = self.augmentation(augmented_data)
        
        # Encode both original and augmented data
        original_encoded = self.encoder(original_data)  # (batch_size, seq_len, d_model)
        augmented_encoded = self.encoder(augmented_data)  # (batch_size, seq_len, d_model)
        
        # Project for contrastive learning
        # Get projections for contrastive learning (only if using contrastive)
        if self.use_contrastive:
            original_projection = self.projection_mlp(original_encoded)  # (batch_size, seq_len, projection_dim)
            augmented_projection = self.projection_mlp(augmented_encoded)  # (batch_size, seq_len, projection_dim)
        else:
            original_projection = None
            augmented_projection = None
        
        # Reconstruct original data from augmented encoding (MLP decoder only needs augmented_encoded)
        reconstructed = self.decoder(augmented_encoded)  # (batch_size, seq_len, input_dim)
        
        return {
            'original_encoded': original_encoded,
            'augmented_encoded': augmented_encoded,
            'original_projection': original_projection,
            'augmented_projection': augmented_projection,
            'reconstructed': reconstructed
        }
    
    def compute_contrastive_loss(self, 
                                original_projection: torch.Tensor, 
                                augmented_projection: torch.Tensor,
                                labels: Optional[torch.Tensor] = None,
                                epsilon: float = 1e-5) -> torch.Tensor:
        """
        Compute weighted contrastive loss (no temperature scaling)
        
        Formula:
        L_Ci = -(1/n) * Σ_{i=1}^{n} log(exp(sim(α_i, α'_i)) / (exp(sim(α_i, α'_i)) + Σ_{j≠i} (1 - sim(α_i, α_j) + ε) * exp(sim(α_i, α'_j))))
        
        Where sim(α_i, α'_i) ∈ [0, 1] (cosine similarity converted to [0,1] range)
        
        Args:
            original_projection: Projected original data (batch_size, seq_len, projection_dim)
            augmented_projection: Projected augmented data (batch_size, seq_len, projection_dim)
            labels: Optional labels for supervised contrastive learning
            epsilon: Small constant for numerical stability (default: 1e-5)
        
        Returns:
            Contrastive loss
        """
        batch_size, seq_len, proj_dim = original_projection.shape
        
        # For contrastive learning, we compare entire windows (sequences)
        # Each window is represented by its mean projection
        orig_flat = torch.mean(original_projection, dim=1)  # (batch_size, proj_dim)
        aug_flat = torch.mean(augmented_projection, dim=1)  # (batch_size, proj_dim)
        n = batch_size
        
        # Normalize projections with eps for numerical stability
        orig_flat = F.normalize(orig_flat, dim=1, eps=1e-8)
        aug_flat = F.normalize(aug_flat, dim=1, eps=1e-8)
        
        # Compute cosine similarity matrices (range: [-1, 1])
        pos_similarity_raw = torch.sum(orig_flat * aug_flat, dim=1)  # (n,)
        neg_similarity_matrix_raw = torch.matmul(orig_flat, orig_flat.T)  # (n, n)
        cross_similarity_matrix_raw = torch.matmul(orig_flat, aug_flat.T)  # (n, n)
        
        # Convert similarity from [-1, 1] to [0, 1] range
        pos_similarity = (pos_similarity_raw + 1) / 2  # (n,)
        neg_similarity_matrix = (neg_similarity_matrix_raw + 1) / 2  # (n, n)
        cross_similarity_matrix = (cross_similarity_matrix_raw + 1) / 2  # (n, n)
     
        # Initialize loss
        total_loss = 0.0
        
        for i in range(n):
            # Get similarity values for sample i
            pos_sim = pos_similarity[i]  # Positive similarity
            neg_weights = 1 - neg_similarity_matrix[i] + epsilon  # (n,)
            neg_weights[i] = 0  # Exclude self (j≠i)
            
            # Clamp neg_weights to be non-negative to avoid negative contributions
            neg_weights = torch.clamp(neg_weights, min=0.0)
            
            cross_sims = cross_similarity_matrix[i]  # Cross similarities
            
            # Log-sum-exp trick to avoid overflow
            # We want to compute: log(exp(pos_sim) + Σ_j neg_weights[j] * exp(cross_sims[j]))
            
            # Find maximum value for numerical stability
            all_sims = torch.cat([pos_sim.unsqueeze(0), cross_sims])
            max_val = torch.max(all_sims)
            
            # Compute log-sum-exp: log(Σ exp(x_i)) = max_val + log(Σ exp(x_i - max_val))
            pos_term = torch.exp(pos_sim - max_val)
            neg_terms = neg_weights * torch.exp(cross_sims - max_val)
            neg_sum = torch.sum(neg_terms)
            
            # Log of denominator using log-sum-exp
            denominator_sum = pos_term + neg_sum + 1e-8
            log_denominator = max_val + torch.log(denominator_sum)
            
            # Compute loss: -log(exp(pos_sim) / denominator) = log_denominator - pos_sim
            loss_i = log_denominator - pos_sim
                
            total_loss += loss_i
        
        # Average loss
        avg_loss = total_loss / n
        
        return avg_loss
    
    def compute_reconstruction_loss(self, 
                                  original_data: torch.Tensor, 
                                  reconstructed_data: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss (MSE)
        
        Args:
            original_data: Original data tensor
            reconstructed_data: Reconstructed data tensor
        
        Returns:
            Reconstruction loss
        """
        return F.mse_loss(reconstructed_data, original_data)
    
    def compute_total_loss(self, 
                          original_data: torch.Tensor, 
                          augmented_data: torch.Tensor,
                          contrastive_weight: float = 1.0,
                          reconstruction_weight: float = 1.0,
                          labels: Optional[torch.Tensor] = None,
                          epsilon: float = 1e-5) -> Dict[str, torch.Tensor]:
        """
        Compute total loss combining contrastive and reconstruction losses
        
        Args:
            original_data: Original data tensor
            augmented_data: Augmented data tensor
            contrastive_weight: Weight for contrastive loss
            reconstruction_weight: Weight for reconstruction loss
            labels: Optional labels for supervised contrastive learning
            epsilon: Small constant for numerical stability in contrastive loss
        
        Returns:
            Dictionary containing individual and total losses
        """
        # Forward pass
        outputs = self.forward(original_data, augmented_data)
        
        # Compute contrastive loss (only if using contrastive)
        if self.use_contrastive:
            contrastive_loss = self.compute_contrastive_loss(
                outputs['original_projection'],
                outputs['augmented_projection'],
                labels,
                epsilon
            )
        else:
            contrastive_loss = torch.tensor(0.0, device=original_data.device)
        
        # Compute reconstruction loss
        reconstruction_loss = self.compute_reconstruction_loss(
            original_data,
            outputs['reconstructed']
        )
        
        # Total loss
        total_loss = (contrastive_weight * contrastive_loss + 
                     reconstruction_weight * reconstruction_loss)
        
        return {
            'total_loss': total_loss,
            'contrastive_loss': contrastive_loss,
            'reconstruction_loss': reconstruction_loss
        }


class WindowSampler:
    """Window sampler for non-overlapping windows with random sampling"""
    
    def __init__(self, data: np.ndarray, window_size: int, stride: int = None):
        """
        Args:
            data: Input data of shape (features, time_steps)
            window_size: Size of each window
            stride: Stride between windows (default: window_size for non-overlapping)
        """
        self.data = data
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size
        
        # Calculate number of windows
        self.n_windows = (data.shape[1] - window_size) // self.stride + 1
        
        # Create window indices
        self.window_indices = []
        for i in range(self.n_windows):
            start_idx = i * self.stride
            end_idx = start_idx + window_size
            self.window_indices.append((start_idx, end_idx))
    
    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample a batch of windows with random augmentation
        
        Args:
            batch_size: Number of windows to sample
        
        Returns:
            Tuple of (original_windows, augmented_windows)
        """
        # Randomly sample window indices
        rng = np.random.default_rng(42)
        sampled_indices = rng.choice(len(self.window_indices), batch_size, replace=True)
        
        original_windows = []
        augmented_windows = []
        
        for idx in sampled_indices:
            start_idx, end_idx = self.window_indices[idx]
            
            # Extract original window
            original_window = self.data[:, start_idx:end_idx]  # (features, window_size)
            
            # Apply random augmentation
            augmented_window = self._apply_random_augmentation(original_window)
            
            original_windows.append(original_window)
            augmented_windows.append(augmented_window)
        
        # Convert to numpy arrays and transpose to (batch_size, window_size, features)
        original_windows = np.array(original_windows).transpose(0, 2, 1)
        augmented_windows = np.array(augmented_windows).transpose(0, 2, 1)
        
        return original_windows, augmented_windows
    
    def _apply_random_augmentation(self, window: np.ndarray) -> np.ndarray:
        """Apply random augmentation to a window - augmentation handled by model"""
        # Augmentation is now handled by the contrastive model's augmentation module
        return window


class ContrastiveDataset(torch.utils.data.Dataset):
    """Dataset for contrastive learning with window sampling"""
    
    def __init__(self,
                 data: np.ndarray,
                 window_size: int,
                 stride: int = None,
                 mask_mode: str = 'none',
                 mask_ratio: float = 0.0,
                 mask_seed: int = None):
        """
        Args:
            data: Input data of shape (features, time_steps)
            window_size: Size of each window
            stride: Stride between windows (default: window_size for non-overlapping)
        """
        self.data = data
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size
        # Masking configuration
        self.mask_mode = mask_mode  # 'none' | 'time' | 'feature'
        self.mask_ratio = float(mask_ratio) if mask_ratio is not None else 0.0
        self.mask_seed = mask_seed
        
        # Create window sampler
        self.sampler = WindowSampler(data, window_size, stride)
        
        # Calculate number of samples
        self.n_samples = self.sampler.n_windows
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # Get window indices
        start_idx, end_idx = self.sampler.window_indices[idx]
        
        # Extract original window
        original_window = self.data[:, start_idx:end_idx]  # (features, window_size)
        
        # Apply masking to ORIGINAL window (masked window is treated as original data)
        if self.mask_mode != 'none' and self.mask_ratio > 0:
            ws = original_window.shape[1]
            feat = original_window.shape[0]
            rng = np.random.RandomState(self.mask_seed + idx) if self.mask_seed is not None else np.random
            if self.mask_mode == 'time':
                num_mask = int(max(1, round(ws * self.mask_ratio)))
                mask_idx = rng.choice(ws, size=min(num_mask, ws), replace=False)
                original_window[:, mask_idx] = 0.0
            elif self.mask_mode == 'feature':
                num_mask = int(max(1, round(feat * self.mask_ratio)))
                mask_idx = rng.choice(feat, size=min(num_mask, feat), replace=False)
                original_window[mask_idx, :] = 0.0

        # For contrastive forward, start augmented as a copy; model's augmentation will change it
        augmented_window = original_window.copy()
        
        # Transpose to (window_size, features)
        original_window = original_window.T
        augmented_window = augmented_window.T
        
        # Convert to tensors
        original_tensor = torch.FloatTensor(original_window)
        augmented_tensor = torch.FloatTensor(augmented_window)
        
        return original_tensor, augmented_tensor


# Example usage and testing
if __name__ == "__main__":
    # Test parameters
    batch_size = 16
    seq_len = 100
    input_dim = 128
    d_model = 256
    projection_dim = 128
    
    # Create sample data
    rng = np.random.default_rng(42)
    data = rng.standard_normal((input_dim, 1000))  # 1000 time steps
    
    # Create model
    model = ContrastiveModel(
        input_dim=input_dim,
        d_model=d_model,
        projection_dim=projection_dim
    )
    
    # Create dataset
    dataset = ContrastiveDataset(data, window_size=seq_len, stride=seq_len)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Test forward pass
    for original_batch, augmented_batch in dataloader:
        print(f"Original batch shape: {original_batch.shape}")
        print(f"Augmented batch shape: {augmented_batch.shape}")
        
        # Forward pass
        outputs = model(original_batch, augmented_batch)
        print(f"Original encoded shape: {outputs['original_encoded'].shape}")
        print(f"Augmented encoded shape: {outputs['augmented_encoded'].shape}")
        print(f"Original projection shape: {outputs['original_projection'].shape}")
        print(f"Augmented projection shape: {outputs['augmented_projection'].shape}")
        print(f"Reconstructed shape: {outputs['reconstructed'].shape}")
        
        # Compute losses
        losses = model.compute_total_loss(original_batch, augmented_batch)
        print(f"Total loss: {losses['total_loss'].item():.4f}")
        print(f"Contrastive loss: {losses['contrastive_loss'].item():.4f}")
        print(f"Reconstruction loss: {losses['reconstruction_loss'].item():.4f}")
        
        break
