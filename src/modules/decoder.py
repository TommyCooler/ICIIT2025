import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer


class TransformerDecoderBlock(nn.Module):
    """Transformer Decoder Block"""
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, dropout=0.1):
        super(TransformerDecoderBlock, self).__init__()
        
        # Transformer decoder layers
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)
        
    def forward(self, tgt, memory):
        """
        Args:
            tgt: Target tensor of shape (batch_size, seq_len, d_model)
            memory: Memory tensor from encoder of shape (batch_size, seq_len, d_model)
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        return self.transformer_decoder(tgt, memory)


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
                nn.ReLU(),
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
        for i in range(0, len(self.tcn_layers), 3):  # Every 3 layers (ConvTranspose1d, ReLU, Dropout)
            conv = self.tcn_layers[i]
            relu = self.tcn_layers[i + 1]
            dropout = self.tcn_layers[i + 2]
            
            # Calculate padding to crop (mirror encoder's TCN behavior)
            dilation = 2 ** (i // 3)
            padding = (self.kernel_size - 1) * dilation
            
            x = conv(x)
            # Crop to maintain sequence length
            if padding > 0:
                x = x[:, :, :-padding]
            x = relu(x)
            x = dropout(x)
        
        # Transpose back: (batch_size, seq_len, output_dim)
        return x.transpose(1, 2)


class Decoder(nn.Module):
    """Decoder with Transformer and TCN blocks for reconstruction"""
    def __init__(self, 
                 d_model, 
                 output_dim,
                 nhead=8, 
                 dim_feedforward=512, 
                 transformer_layers=6,
                 tcn_output_dim=None,
                 tcn_kernel_size=3,
                 tcn_num_layers=3,
                 dropout=0.1,
                 combination_method='concat'):
        """
        Args:
            d_model: Model dimension from encoder
            output_dim: Output dimension (should match input dimension of original data)
            nhead: Number of attention heads
            dim_feedforward: Feedforward dimension
            transformer_layers: Number of transformer decoder layers
            tcn_output_dim: Output dimension for TCN (default: same as d_model)
            tcn_kernel_size: Kernel size for TCN
            tcn_num_layers: Number of TCN layers
            dropout: Dropout rate
            combination_method: 'concat' or 'stack' for combining outputs
        """
        super(Decoder, self).__init__()
        
        # Transformer Decoder Block
        self.transformer_block = TransformerDecoderBlock(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=transformer_layers,
            dropout=dropout
        )
        
        # TCN Decoder Block
        if tcn_output_dim is None:
            tcn_output_dim = d_model
            
        self.tcn_block = TCNDecoderBlock(
            input_dim=d_model,
            output_dim=tcn_output_dim,
            kernel_size=tcn_kernel_size,
            num_layers=tcn_num_layers,
            dropout=dropout
        )
        
        # Combination method
        self.combination_method = combination_method
        
        # Output projection
        if combination_method == 'concat':
            self.output_projection = nn.Linear(d_model + tcn_output_dim, output_dim)
        elif combination_method == 'stack':
            # For stack, we need to handle the extra dimension
            self.output_projection = nn.Linear(d_model, output_dim)
        else:
            raise ValueError("combination_method must be 'concat' or 'stack'")
    
    def forward(self, x, memory):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model) - encoded augmented data
            memory: Memory tensor from encoder of shape (batch_size, seq_len, d_model) - encoded original data
        Returns:
            Reconstructed output tensor of shape (batch_size, seq_len, output_dim)
        """
        # Transformer Decoder (uses memory from encoder)
        transformer_output = self.transformer_block(x, memory)  # (batch_size, seq_len, d_model)
        
        # TCN Decoder (uses input x)
        tcn_output = self.tcn_block(x)  # (batch_size, seq_len, tcn_output_dim)
        
        # Combine outputs
        if self.combination_method == 'concat':
            # Concatenate along feature dimension
            combined = torch.cat([transformer_output, tcn_output], dim=-1)
            # Project to output dimension
            output = self.output_projection(combined)
            return output
            
        elif self.combination_method == 'stack':
            # For stack, use transformer output and project to output dimension
            output = self.output_projection(transformer_output)
            return output
    
    def get_individual_outputs(self, x, memory):
        """
        Get outputs from individual blocks for analysis
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            memory: Memory tensor from encoder
        Returns:
            Dictionary with transformer and TCN outputs
        """
        with torch.no_grad():
            transformer_output = self.transformer_block(x, memory)
            tcn_output = self.tcn_block(x)
            
            return {
                'transformer': transformer_output,
                'tcn': tcn_output
            }


# Example usage and testing
if __name__ == "__main__":
    # Test parameters
    batch_size = 32
    seq_len = 100
    d_model = 256
    output_dim = 128
    
    print("Testing Decoder with concat method:")
    decoder_concat = Decoder(
        d_model=d_model,
        output_dim=output_dim,
        nhead=8,
        transformer_layers=6,
        tcn_output_dim=128,
        combination_method='concat'
    )
    
    x = torch.randn(batch_size, seq_len, d_model)  # Encoded augmented data
    memory = torch.randn(batch_size, seq_len, d_model)  # Encoded original data
    output_concat = decoder_concat(x, memory)
    print(f"Input shape: {x.shape}")
    print(f"Memory shape: {memory.shape}")
    print(f"Output shape (concat): {output_concat.shape}")
    
    print("\nTesting Decoder with stack method:")
    decoder_stack = Decoder(
        d_model=d_model,
        output_dim=output_dim,
        nhead=8,
        transformer_layers=6,
        tcn_output_dim=d_model,  # Same as d_model for stack
        combination_method='stack'
    )
    
    output_stack = decoder_stack(x, memory)
    print(f"Output shape (stack): {output_stack.shape}")
    
    print("\nIndividual outputs:")
    individual_outputs = decoder_concat.get_individual_outputs(x, memory)
    for name, output_tensor in individual_outputs.items():
        print(f"{name}: {output_tensor.shape}")
