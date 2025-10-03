import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPDecoderBlock(nn.Module):
    """Simple MLP Decoder Block for reconstruction"""
    def __init__(self, d_model, output_dim, hidden_dims=None, dropout=0.1):
        super(MLPDecoderBlock, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [d_model, d_model // 2, d_model // 4]
        
        # Build MLP layers
        layers = []
        input_dim = d_model
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(input_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Output tensor of shape (batch_size, seq_len, output_dim)
        """
        return self.mlp(x)


class TCNDecoderBlock(nn.Module):
    """Temporal Convolutional Network Decoder Block"""
    def __init__(self, input_dim, output_dim, kernel_size=3, num_layers=3, dropout=0.1):
        super(TCNDecoderBlock, self).__init__()
        
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
        # Transpose for ConvTranspose1d: (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Apply TCN layers
        for i in range(0, len(self.tcn_layers), 3):  # Every 3 layers (ConvTranspose1d, GELU, Dropout)
            conv = self.tcn_layers[i]
            gelu = self.tcn_layers[i + 1]
            dropout = self.tcn_layers[i + 2]
            
            # Calculate padding to crop (mirror encoder's TCN behavior)
            dilation = 2 ** (i // 3)
            padding = (self.kernel_size - 1) * dilation
            
            x = conv(x)
            # Crop to maintain sequence length
            if padding > 0:
                x = x[:, :, :-padding]
            x = gelu(x)
            x = dropout(x)
        
        # Transpose back: (batch_size, seq_len, output_dim)
        return x.transpose(1, 2)


class Decoder(nn.Module):
    """Simple MLP Decoder for reconstruction"""
    def __init__(self, 
                 d_model, 
                 output_dim,
                 hidden_dims=None,
                 dropout=0.1):
        """
        Args:
            d_model: Model dimension from encoder
            output_dim: Output dimension (should match input dimension of original data)
            hidden_dims: List of hidden dimensions for MLP (default: [d_model, d_model//2, d_model//4])
            dropout: Dropout rate
        """
        super(Decoder, self).__init__()
        
        # Simple MLP Decoder Block
        self.mlp_block = MLPDecoderBlock(
            d_model=d_model,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model) - encoded augmented data
        Returns:
            Reconstructed output tensor of shape (batch_size, seq_len, output_dim)
        """
        # Simple MLP reconstruction (no memory, no complex architecture)
        output = self.mlp_block(x)
        return output
    
    def get_individual_outputs(self, x):
        """
        Get outputs for analysis (for compatibility)
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Dictionary with MLP output
        """
        with torch.no_grad():
            mlp_output = self.mlp_block(x)
            
            return {
                'mlp': mlp_output
            }


# Example usage and testing
if __name__ == "__main__":
    # Test parameters
    batch_size = 32
    seq_len = 100
    d_model = 256
    output_dim = 128
    
    print("Testing MLP Decoder:")
    decoder = Decoder(
        d_model=d_model,
        output_dim=output_dim,
        hidden_dims=[d_model, d_model//2, d_model//4],
        dropout=0.1
    )
    
    x = torch.randn(batch_size, seq_len, d_model)  # Encoded augmented data
    output = decoder(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    print("\nIndividual outputs:")
    individual_outputs = decoder.get_individual_outputs(x)
    for name, output_tensor in individual_outputs.items():
        print(f"{name}: {output_tensor.shape}")
    
    print(f"\nModel parameters: {sum(p.numel() for p in decoder.parameters()):,}")
