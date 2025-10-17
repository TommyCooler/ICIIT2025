import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import custom modules
from src.utils.connv1d import TemporalConv1d

class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding for Transformer"""
    def __init__(self, d_model: int, window_size: int = 5000):
        """
        Args:
            d_model: Model dimension
            window_size: Window size for positional encoding
        """
        super(LearnablePositionalEncoding, self).__init__()
        self.d_model = d_model
        self.window_size = window_size
        
        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.zeros(window_size, d_model))
        
        # Initialize with small random values
        # LearnablePositionalEncoding.__init__
        nn.init.normal_(self.pos_embedding, mean=0.0, std=(1.0 / math.sqrt(d_model)))

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Output tensor with positional encoding added
        """
        _, seq_len, _ = x.shape
        
        # Ensure we don't exceed window_size
        if seq_len > self.window_size:
            raise ValueError(f"Sequence length {seq_len} exceeds window size {self.window_size}")
        
        # Get positional embeddings for current sequence length
        pos_emb = self.pos_embedding[:seq_len, :]  # (seq_len, d_model)
        
        # Add positional encoding to input
        return x + pos_emb.unsqueeze(0)  # (batch_size, seq_len, d_model)


class LinearAugmentation(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.weights1 = nn.Parameter(torch.empty(output_dim, input_dim))
        nn.init.normal_(self.weights1, mean=0.0, std=0.001)
        # bias theo kênh, broadcast theo T
        self.bias1 = nn.Parameter(torch.zeros(output_dim, 1))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x_ft = x.transpose(1, 2)                       # (B,F_in,T)
        y = torch.matmul(self.weights1, x_ft) + self.bias1  # (B,F_out,T) broadcast
        y = self.drop(y).transpose(1, 2)               # (B,T,F_out)
        return y

class MLPAugmentation(nn.Module):
    """MLP Augmentation with feature flattening"""
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout)
        
        # Store MLP layers for different flattened sizes
        self._mlp_cache = {}

    def forward(self, x):  # (B,T,F)
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Output tensor of shape (batch_size, seq_len, output_dim)
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Flatten features: (batch_size, seq_len, input_dim) -> (batch_size, seq_len * input_dim)
        x_flattened = x.view(batch_size, seq_len * input_dim)
        flattened_size = seq_len * input_dim
        output_flattened_size = seq_len * self.output_dim
        
        # Create or get cached MLP for this flattened size
        if flattened_size not in self._mlp_cache:
            dropout_rate = self.dropout.p if hasattr(self.dropout, 'p') else 0.1
            self._mlp_cache[flattened_size] = nn.Sequential(
                nn.Linear(flattened_size, flattened_size * 2),  # Expand features
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(flattened_size * 2, flattened_size),  # Compress back
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(flattened_size, output_flattened_size)  # Final output dimension
            ).to(x.device)
        
        mlp = self._mlp_cache[flattened_size]
        
        # Apply MLP to flattened features
        # (batch_size, seq_len * input_dim) -> (batch_size, seq_len * output_dim)
        output_flattened = mlp(x_flattened)
        
        # Reshape back: (batch_size, seq_len * output_dim) -> (batch_size, seq_len, output_dim)
        output = output_flattened.view(batch_size, seq_len, self.output_dim)
        
        return output

class CNNAugmentation(nn.Module):
    """
    1D CNN augmentation (giữ nguyên T)
    - Padding đều 2 bên (non-causal 'same' đối xứng)
    """
    def __init__(self, input_dim, output_dim, kernel_size=3, dropout=0.1,
                 dilation=1):
        super().__init__()
        self.conv = TemporalConv1d(
            input_dim, output_dim, kernel_size,
            dilation=dilation, causal=False, pad_mode="zeros"
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # (B,T,C) hoặc (T,C)
        single = (x.dim() == 2)
        if single: x = x.unsqueeze(0)
        x = x.transpose(1, 2)              # (B,C,T)
        x = self.conv(x)                   # giữ nguyên T
        x = F.gelu(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)              # (B,T,C_out)
        return x.squeeze(0) if single else x

class TCNAugmentation(nn.Module):
    """
    TCN augmentation (stack conv1d, giữ nguyên T)
    - Padding bên trái (causal - không rò rỉ tương lai)
    dilation: int hoặc list[int] (vd: [1,2,4,8,...] để tăng receptive field)
    """
    def __init__(self, input_dim, output_dim, kernel_size=3, num_layers=1, dropout=0.1,
                 dilation=1):
        super().__init__()
        L = max(1, int(num_layers))
        if isinstance(dilation, int):
            dilations = [dilation] * L
        else:
            dilations = list(dilation)
            assert len(dilations) == L, "len(dilation) phải bằng num_layers"

        layers, in_ch = [], input_dim
        for i in range(L):
            out_ch = output_dim if i == L - 1 else input_dim
            layers += [
                TemporalConv1d(
                    in_ch, out_ch, kernel_size,
                    dilation=dilations[i], causal=True, pad_mode="zeros"
                ),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # (B,T,C) hoặc (T,C)
        single = (x.dim() == 2)
        if single: x = x.unsqueeze(0)
        x = x.transpose(1, 2)              # (B,C,T)
        x = self.net(x)                    # giữ nguyên T
        x = x.transpose(1, 2)              # (B,T,C_out)
        return x.squeeze(0) if single else x




class LSTMAugmentation(nn.Module):
    """LSTM for augmentation"""
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super(LSTMAugmentation, self).__init__()
        self.lstm = nn.LSTM(input_dim, output_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim) hoặc (seq_len, input_dim)
        if x.dim() == 2:
            # x shape: (seq_len, input_dim) -> add batch dimension
            x = x.unsqueeze(0)  # (1, seq_len, input_dim)
            x, _ = self.lstm(x)
            result = self.dropout(x)
            return result.squeeze(0)  # (seq_len, output_dim)
        else:
            # x shape: (batch_size, seq_len, input_dim)
            x, _ = self.lstm(x)
            return self.dropout(x)


class EncoderTransformerAugmentation(nn.Module):
    def __init__(self, input_dim, output_dim, nhead=8, num_layers=1, dropout=0.1, window_size=5000):
        super().__init__()
        if input_dim < nhead:
            nhead = max(1, input_dim)

        if output_dim % nhead != 0:
            output_dim = ((output_dim + nhead - 1) // nhead) * nhead

        self.input_projection = nn.Linear(input_dim, output_dim)
        self.pos_encoding = LearnablePositionalEncoding(output_dim, window_size)

        layer = TransformerEncoderLayer(
            d_model=output_dim, nhead=nhead,
            dim_feedforward=max(output_dim * 4, 64),
            dropout=dropout, 
            batch_first=True,
            norm_first=True
        )
        self.encoder = TransformerEncoder(layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # (B,T,F) or (T,F)
        single = (x.dim() == 2)
        if single: x = x.unsqueeze(0)
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        x = self.dropout(x)
        return x.squeeze(0) if single else x



class Augmentation(nn.Module):
    """
    Main augmentation class that combines all nonlinear modules
    
    Supports various input dimensions:
    - gesture: 2 features (x, y coordinates)
    - pd: 1 feature (univariate time series)
    - ecg: 2 features (2-lead ECG)
    - psm: 25 features (multivariate sensors)
    - nab: 1 feature (univariate time series)
    - smap_msl: 25 features (multivariate sensors)
    - smd: 38 features (multivariate sensors)
    - ucr: 1 feature (univariate time series)
    """
    def __init__(self, input_dim, output_dim, dropout=0.1, temperature=1.0, window_size=5000, **kwargs):
        super(Augmentation, self).__init__()
        
        # Get nhead for transformer - adjust for small input dimensions
        nhead = kwargs.get('nhead', 8)
        if input_dim < nhead:
            nhead = max(1, input_dim)  # Use input_dim as nhead if smaller
        
        # Temperature parameter for softmax (tau)
        self.temperature = temperature
        
        # Enforce identical input/output feature dimension for all modules
        desired_output_dim = input_dim
        
        # Ensure output_dim is divisible by nhead for transformer
        transformer_output_dim = desired_output_dim
        if desired_output_dim % nhead != 0:
            transformer_output_dim = ((desired_output_dim + nhead - 1) // nhead) * nhead
        
        # Initialize all augmentation modules
        self.linear_module = LinearAugmentation(input_dim, desired_output_dim, dropout)
        self.mlp_module = MLPAugmentation(input_dim, desired_output_dim, dropout)
        self.cnn_module = CNNAugmentation(input_dim, desired_output_dim, 
                                        kernel_size=kwargs.get('cnn_kernel_size', 3), 
                                        dropout=dropout)
        self.tcn_module = TCNAugmentation(
            input_dim,
            desired_output_dim,
            kernel_size=kwargs.get('tcn_kernel_size', 3),
            num_layers=kwargs.get('tcn_num_layers', 1),
            dropout=dropout
        )
        self.lstm_module = LSTMAugmentation(input_dim, desired_output_dim, dropout)
        self.transformer_module = EncoderTransformerAugmentation(
            input_dim, transformer_output_dim,
            nhead=nhead,
            num_layers=kwargs.get('num_layers', 1),
            dropout=dropout,
            window_size=window_size
        )
        
        # Add projection layer if transformer output dim is different
        if transformer_output_dim != desired_output_dim:
            self.transformer_projection = nn.Linear(transformer_output_dim, desired_output_dim)
        else:
            self.transformer_projection = None
        
        # Weight parameters for combining outputs
        self.alpha = nn.Parameter(torch.ones(6) / 6)  # 6 modules
        
    def forward(self, x):
        """
        Forward pass through all augmentation modules and combine results
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Combined output of shape (batch_size, seq_len, output_dim)
        """
        # Get outputs from all modules
        linear_out = self.linear_module(x)
        mlp_out = self.mlp_module(x)
        cnn_out = self.cnn_module(x)
        tcn_out = self.tcn_module(x)
        lstm_out = self.lstm_module(x)
        transformer_out = self.transformer_module(x)
        
        # Apply projection if needed
        if self.transformer_projection is not None:
            transformer_out = self.transformer_projection(transformer_out)
        
        # Stack all outputs: (num_aug, batch, seq_len, feat)
        outputs = torch.stack([linear_out, mlp_out, cnn_out, tcn_out, lstm_out, transformer_out], dim=0)

        # # Learned probabilities for 6 augmentations (after temperature-scaled gumbel-softmax)
        # probs = F.gumbel_softmax(self.alpha, tau=self.temperature, hard=False, dim=0)  # (6,)

        # # Flatten per augmentation, apply weights, reshape back, then sum over aug dimension
        # num_aug, bsz, seq_len, feat = outputs.shape
        # outputs_flat = outputs.reshape(num_aug, bsz * seq_len * feat)  # (6, N)
        # weighted_flat = torch.unsqueeze(probs, -1) * outputs_flat       # (6, N)
        # weighted = weighted_flat.reshape(num_aug, bsz, seq_len, feat)   # (6, B, T, D)
        # combined_output = torch.sum(weighted, dim=0)                    # (B, T, D)
        
        probs = F.gumbel_softmax(self.alpha, tau=self.temperature, hard=True, dim=0)  # (6,)
        weighted = outputs * probs.view(-1, 1, 1, 1)                      # broadcast
        combined_output = weighted.sum(dim=0)  
        
        return combined_output
    
