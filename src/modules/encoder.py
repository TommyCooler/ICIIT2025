import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding for Transformer"""
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super(LearnablePositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.zeros(max_len, d_model))
        
        # Initialize with small random values
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Output tensor with positional encoding added
        """
        batch_size, seq_len, d_model = x.shape
        
        # Ensure we don't exceed max_len
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.max_len}")
        
        # Get positional embeddings for current sequence length
        pos_emb = self.pos_embedding[:seq_len, :]  # (seq_len, d_model)
        
        # Add positional encoding to input
        return x + pos_emb.unsqueeze(0)  # (batch_size, seq_len, d_model)


class TransformerEncoderBlock(nn.Module):
    """Transformer Encoder Block with learnable positional encoding"""
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, dropout=0.1, max_len=5000):
        super(TransformerEncoderBlock, self).__init__()
        
        # Learnable positional encoding
        self.pos_encoding = LearnablePositionalEncoding(d_model, max_len)
        
        # Transformer encoder layers
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer encoder
        return self.transformer_encoder(x)


class TCNBlock(nn.Module):
    """Temporal Convolutional Network Block"""
    def __init__(self, input_dim, output_dim, kernel_size=3, num_layers=3, dropout=0.1):
        super(TCNBlock, self).__init__()
        
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i  # Exponential dilation
            in_channels = input_dim if i == 0 else output_dim
            
            layers.extend([
                nn.Conv1d(
                    in_channels, 
                    output_dim, 
                    kernel_size, 
                    padding=(kernel_size-1)*dilation, 
                    dilation=dilation
                ),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
        
        self.tcn_layers = nn.ModuleList(layers)
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Output tensor of shape (batch_size, seq_len, output_dim)
        """
        # Transpose for Conv1d: (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Apply TCN layers
        for i in range(0, len(self.tcn_layers), 3):  # Every 3 layers (Conv1d, GELU, Dropout)
            conv = self.tcn_layers[i]
            gelu = self.tcn_layers[i + 1]
            dropout = self.tcn_layers[i + 2]
            
            # Calculate padding to crop
            dilation = 2 ** (i // 3)
            padding = (self.kernel_size - 1) * dilation
            
            x = conv(x)
            # Crop to maintain sequence length (crop from end to keep causal behavior)
            if padding > 0:
                x = x[:, :, :-padding]
            x = gelu(x)
            x = dropout(x)
        
        # Transpose back: (batch_size, seq_len, output_dim)
        return x.transpose(1, 2)


class Encoder(nn.Module):
    """Encoder with Transformer and TCN blocks"""
    def __init__(self, 
                 input_dim, 
                 d_model, 
                 nhead=8, 
                 dim_feedforward=512, 
                 transformer_layers=6,
                 tcn_output_dim=None,
                 tcn_kernel_size=2,
                 tcn_num_layers=4,
                 dropout=0.1,
                 combination_method='concat',
                 max_len=5000):
        """
        Args:
            input_dim: Input dimension
            d_model: Model dimension for transformer
            nhead: Number of attention heads
            dim_feedforward: Feedforward dimension
            transformer_layers: Number of transformer encoder layers
            tcn_output_dim: Output dimension for TCN (default: same as d_model)
            tcn_kernel_size: Kernel size for TCN
            tcn_num_layers: Number of TCN layers
            dropout: Dropout rate
            combination_method: 'concat' or 'stack' for combining outputs
            max_len: Maximum sequence length for positional encoding
        """
        super(Encoder, self).__init__()
        
        # Input projection to d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Transformer Encoder Block
        self.transformer_block = TransformerEncoderBlock(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=transformer_layers,
            dropout=dropout,
            max_len=max_len
        )
        
        # TCN Block
        if tcn_output_dim is None:
            tcn_output_dim = d_model
            
        self.tcn_block = TCNBlock(
            input_dim=input_dim,  # TCN takes original input
            output_dim=tcn_output_dim,
            kernel_size=tcn_kernel_size,
            num_layers=tcn_num_layers,
            dropout=dropout
        )
        
        # Combination method
        self.combination_method = combination_method
        
        # Output projection if needed
        if combination_method == 'concat':
            self.output_projection = nn.Linear(d_model + tcn_output_dim, d_model)
        elif combination_method == 'stack':
            # For stack, we will concatenate then project back to d_model
            self.stack_projection = nn.Linear(d_model + tcn_output_dim, d_model)
        else:
            raise ValueError("combination_method must be 'concat' or 'stack'")
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Combined output tensor
        """
        # Project input to d_model for transformer
        x_projected = self.input_projection(x)
        
        # Transformer Encoder
        transformer_output = self.transformer_block(x_projected)  # (batch_size, seq_len, d_model)
        
        # TCN (uses original input)
        tcn_output = self.tcn_block(x)  # (batch_size, seq_len, tcn_output_dim)
        
        # Combine outputs
        if self.combination_method == 'concat':
            # Concatenate along feature dimension
            combined = torch.cat([transformer_output, tcn_output], dim=-1)
            # Project back to d_model
            output = self.output_projection(combined)
            return output
            
        elif self.combination_method == 'stack':
            # Concatenate features and project back to d_model to keep 3D output
            if tcn_output.size(-1) != transformer_output.size(-1):
                pad_size = transformer_output.size(-1) - tcn_output.size(-1)
                tcn_output = F.pad(tcn_output, (0, pad_size))
            combined = torch.cat([transformer_output, tcn_output], dim=-1)
            output = self.stack_projection(combined)
            return output
    


