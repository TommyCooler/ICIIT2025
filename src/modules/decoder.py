import torch
import torch.nn as nn
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import custom modules
from src.modules.customLinear import CustomLinear


class CustomLinearDecoderBlock(nn.Module):
    """Custom Linear Decoder Block using CustomLinear layer"""
    def __init__(self, d_model, output_dim, dropout=0.1):
        super(CustomLinearDecoderBlock, self).__init__()
        
        self.d_model = d_model
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout)
        
        # We'll create CustomLinear layers dynamically in forward pass
        # since seq_len can vary
        self._custom_linear_cache = {}
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Output tensor of shape (batch_size, seq_len, output_dim)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Create or get cached CustomLinear layer for this seq_len
        if seq_len not in self._custom_linear_cache:
            self._custom_linear_cache[seq_len] = CustomLinear(
                input_shape=(d_model, seq_len),
                output_shape=(self.output_dim, seq_len)
            ).to(x.device)
        
        custom_linear = self._custom_linear_cache[seq_len]
        
        # Reshape for CustomLinear: (batch_size, seq_len, d_model) -> (batch_size, d_model, seq_len)
        x_reshaped = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        
        # Apply CustomLinear to each sample in batch
        outputs = []
        for i in range(batch_size):
            # Extract single sample: (d_model, seq_len)
            sample = x_reshaped[i]
            
            # Apply CustomLinear: (d_model, seq_len) -> (output_dim, seq_len)
            output_sample = custom_linear(sample)
            
            # Transpose back: (output_dim, seq_len) -> (seq_len, output_dim)
            output_sample = output_sample.transpose(0, 1)
            outputs.append(output_sample)
        
        # Stack outputs: (batch_size, seq_len, output_dim)
        output = torch.stack(outputs, dim=0)
        
        # Apply dropout
        output = self.dropout(output)
        
        return output


class Decoder(nn.Module):
    """Simplified Decoder using only CustomLinear"""
    def __init__(self, d_model, output_dim, dropout=0.1):
        """
        Args:
            d_model: Model dimension from encoder (after concat and projection)
            output_dim: Output dimension (should match input dimension of original data)
            dropout: Dropout rate
        """
        super(Decoder, self).__init__()
        
        self.d_model = d_model
        self.output_dim = output_dim
        
        # Only use CustomLinear decoder
        # Input: (batch_size, seq_len, d_model) from encoder
        # Output: (batch_size, seq_len, output_dim) to match original input
        self.decoder_block = CustomLinearDecoderBlock(
            d_model=d_model,
            output_dim=output_dim,
            dropout=dropout
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model) - encoded data from encoder (after concat)
        Returns:
            Reconstructed output tensor of shape (batch_size, seq_len, output_dim) - matches original input shape
        """
        output = self.decoder_block(x)
        return output