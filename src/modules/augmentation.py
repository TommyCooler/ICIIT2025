import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math


class LinearAugmentation(nn.Module):
    """Linear layer for augmentation"""
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super(LinearAugmentation, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim) hoặc (seq_len, input_dim) hoặc (batch_size * seq_len, input_dim)
        if x.dim() == 2:
            # x shape: (seq_len, input_dim) hoặc (batch_size * seq_len, input_dim)
            # For 2D input, we need to handle it as (seq_len, input_dim)
            # Add batch dimension and process
            x = x.unsqueeze(0)  # (1, seq_len, input_dim)
            result = self.dropout(F.relu(self.linear(x)))
            return result.squeeze(0)  # (seq_len, output_dim)
        else:
            # x shape: (batch_size, seq_len, input_dim)
            return self.dropout(F.relu(self.linear(x)))


class CNNAugmentation(nn.Module):
    """1D CNN for augmentation"""
    def __init__(self, input_dim, output_dim, kernel_size=3, dropout=0.1):
        super(CNNAugmentation, self).__init__()
        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size, padding=kernel_size//2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim) hoặc (seq_len, input_dim)
        if x.dim() == 2:
            # x shape: (seq_len, input_dim) -> add batch dimension
            x = x.unsqueeze(0)  # (1, seq_len, input_dim)
            x = x.transpose(1, 2)  # (1, input_dim, seq_len)
            x = self.conv(x)
            x = F.relu(x)
            x = self.dropout(x)
            result = x.transpose(1, 2)  # (1, seq_len, output_dim)
            return result.squeeze(0)  # (seq_len, output_dim)
        else:
            # x shape: (batch_size, seq_len, input_dim)
            x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
            x = self.conv(x)
            x = F.relu(x)
            x = self.dropout(x)
            return x.transpose(1, 2)  # (batch_size, seq_len, output_dim)


class TCNAugmentation(nn.Module):
    """Temporal Convolutional augmentation (single-layer, no dilation)"""
    def __init__(self, input_dim, output_dim, kernel_size=3, dropout=0.1):
        super(TCNAugmentation, self).__init__()
        # No dilation, single conv layer, use same-like padding
        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size, padding=kernel_size//2, dilation=1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim) hoặc (seq_len, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, seq_len, input_dim)
            x = x.transpose(1, 2)  # (1, input_dim, seq_len)
            x = self.conv(x)
            x = F.relu(x)
            x = self.dropout(x)
            result = x.transpose(1, 2)  # (1, seq_len, output_dim)
            return result.squeeze(0)  # (seq_len, output_dim)
        else:
            x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
            x = self.conv(x)
            x = F.relu(x)
            x = self.dropout(x)
            return x.transpose(1, 2)  # (batch_size, seq_len, output_dim)


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
    """Transformer Encoder for augmentation (single layer)"""
    def __init__(self, input_dim, output_dim, nhead=8, num_layers=1, dropout=0.1):
        super(EncoderTransformerAugmentation, self).__init__()
        
        # Ensure output_dim is divisible by nhead
        if output_dim % nhead != 0:
            output_dim = ((output_dim + nhead - 1) // nhead) * nhead
        
        # Linear projection để đảm bảo output_dim
        self.input_projection = nn.Linear(input_dim, output_dim)
        
        # Transformer encoder layer
        encoder_layer = TransformerEncoderLayer(
            d_model=output_dim,
            nhead=nhead,
            dim_feedforward=output_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        # Force to a single layer transformer
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim) hoặc (seq_len, input_dim)
        if x.dim() == 2:
            # x shape: (seq_len, input_dim) -> add batch dimension
            x = x.unsqueeze(0)  # (1, seq_len, input_dim)
            x = self.input_projection(x)  # (1, seq_len, output_dim)
            x = self.transformer_encoder(x)
            result = self.dropout(x)
            return result.squeeze(0)  # (seq_len, output_dim)
        else:
            # x shape: (batch_size, seq_len, input_dim)
            x = self.input_projection(x)  # (batch_size, seq_len, output_dim)
            x = self.transformer_encoder(x)
            return self.dropout(x)


class Augmentation(nn.Module):
    """Main augmentation class that combines all nonlinear modules"""
    def __init__(self, input_dim, output_dim, dropout=0.1, temperature=1.0, **kwargs):
        super(Augmentation, self).__init__()
        
        # Get nhead for transformer
        nhead = kwargs.get('nhead', 8)
        
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
        self.cnn_module = CNNAugmentation(input_dim, desired_output_dim, 
                                        kernel_size=kwargs.get('cnn_kernel_size', 3), 
                                        dropout=dropout)
        self.tcn_module = TCNAugmentation(
            input_dim,
            desired_output_dim,
            kernel_size=kwargs.get('tcn_kernel_size', 3),
            dropout=dropout
        )
        self.lstm_module = LSTMAugmentation(input_dim, desired_output_dim, dropout)
        self.transformer_module = EncoderTransformerAugmentation(
            input_dim, transformer_output_dim,
            nhead=nhead,
            num_layers=kwargs.get('num_layers', 1),
            dropout=dropout
        )
        
        # Add projection layer if transformer output dim is different
        if transformer_output_dim != desired_output_dim:
            self.transformer_projection = nn.Linear(transformer_output_dim, desired_output_dim)
        else:
            self.transformer_projection = None
        
        # Weight parameters for combining outputs
        self.alpha = nn.Parameter(torch.ones(5) / 5)  # 5 modules
        
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
        cnn_out = self.cnn_module(x)
        tcn_out = self.tcn_module(x)
        lstm_out = self.lstm_module(x)
        transformer_out = self.transformer_module(x)
        
        # Apply projection if needed
        if self.transformer_projection is not None:
            transformer_out = self.transformer_projection(transformer_out)
        
        # Stack all outputs: (num_aug, batch, seq_len, feat)
        outputs = torch.stack([linear_out, cnn_out, tcn_out, lstm_out, transformer_out], dim=0)

        # Learned probabilities for 5 augmentations (after temperature-scaled softmax)
        probs = F.softmax(self.alpha / self.temperature, dim=0)  # (5,)

        # Flatten per augmentation, apply weights, reshape back, then sum over aug dimension
        num_aug, bsz, seq_len, feat = outputs.shape
        outputs_flat = outputs.reshape(num_aug, bsz * seq_len * feat)  # (5, N)
        weighted_flat = torch.unsqueeze(probs, -1) * outputs_flat       # (5, N)
        weighted = weighted_flat.reshape(num_aug, bsz, seq_len, feat)   # (5, B, T, D)
        combined_output = torch.sum(weighted, dim=0)                    # (B, T, D)
        
        return combined_output
    
    def get_module_outputs(self, x):
        """
        Get individual outputs from each module for analysis
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Dictionary containing outputs from each module
        """
        with torch.no_grad():
            transformer_out = self.transformer_module(x)
            if self.transformer_projection is not None:
                transformer_out = self.transformer_projection(transformer_out)
            outputs = {
                'linear': self.linear_module(x),
                'cnn': self.cnn_module(x),
                'tcn': self.tcn_module(x),
                'lstm': self.lstm_module(x),
                'transformer': transformer_out
            }
        return outputs
    
    def get_combination_weights(self):
        """Get current combination weights"""
        return F.softmax(self.alpha / self.temperature, dim=0)


# Example usage and testing
if __name__ == "__main__":
    # Test parameters
    batch_size = 32
    seq_len = 100
    input_dim = 128
    output_dim = 64
    
    # Test different temperature values
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    print("Testing Augmentation with different temperatures:")
    print("=" * 60)
    
    for temp in temperatures:
        print(f"\nTemperature: {temp}")
        print("-" * 30)
        
        # Create augmentation module
        augmentation = Augmentation(input_dim, output_dim, dropout=0.1, temperature=temp)
        
        # Create sample input
        x = torch.randn(batch_size, seq_len, input_dim)
        
        # Forward pass
        output = augmentation(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        
        # Get combination weights
        weights = augmentation.get_combination_weights()
        print(f"Combination weights: {weights.detach().numpy()}")
        print(f"Weight entropy: {-torch.sum(weights * torch.log(weights + 1e-8)).item():.4f}")
    
    print("\n" + "=" * 60)
    print("Temperature effects:")
    print("- Low temperature (0.1): Sharp distribution, one augmentation dominates")
    print("- Medium temperature (1.0): Balanced distribution")
    print("- High temperature (5.0): Smooth distribution, more uniform mixing")
