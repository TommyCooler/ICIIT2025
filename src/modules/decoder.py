import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .customLinear import CustomLinear


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
    def __init__(self, d_model, output_dim, kernel_size=3, num_layers=3, dropout=0.1):
        super(TCNDecoderBlock, self).__init__()
        
        # Build TCN layers with residual connections
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i  # Exponential dilation
            in_channels = d_model if i == 0 else d_model
            
            # TCN block with residual connection
            tcn_block = nn.ModuleList([
                nn.Conv1d(
                    in_channels, 
                    d_model, 
                    kernel_size, 
                    padding=(kernel_size-1)*dilation, 
                    dilation=dilation
                ),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            layers.append(tcn_block)
        
        self.tcn_blocks = nn.ModuleList(layers)
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        
        # Output projection layer
        self.output_projection = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Output tensor of shape (batch_size, seq_len, output_dim)
        """
        # Transpose for Conv1d: (batch_size, d_model, seq_len)
        x = x.transpose(1, 2)
        
        # Apply TCN blocks with residual connections
        for i, tcn_block in enumerate(self.tcn_blocks):
            residual = x
            
            # Apply TCN block
            for layer in tcn_block:
                if isinstance(layer, nn.Conv1d):
                    x = layer(x)
                    # Crop to maintain sequence length
                    dilation = 2 ** i
                    padding = (self.kernel_size - 1) * dilation
                    if padding > 0:
                        x = x[:, :, :-padding]
                else:
                    x = layer(x)
            
            # Residual connection
            x = x + residual
        
        # Transpose back: (batch_size, seq_len, d_model)
        x = x.transpose(1, 2)
        
        # Final output projection
        output = self.output_projection(x)
        
        return output


class TransformerDecoderBlock(nn.Module):
    """Transformer Decoder Block with self-attention"""
    def __init__(self, d_model, output_dim, nhead=8, num_layers=3, dim_feedforward=512, dropout=0.1):
        super(TransformerDecoderBlock, self).__init__()
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output projection layer
        self.output_projection = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Output tensor of shape (batch_size, seq_len, output_dim)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Create memory (same as input for self-attention)
        memory = x
        
        # Create target sequence (same as input for self-attention)
        tgt = x
        
        # Apply transformer decoder (self-attention)
        # Note: We use the same tensor as both target and memory for self-attention
        output = self.transformer_decoder(tgt, memory)
        
        # Final output projection
        output = self.output_projection(output)
        
        return output


class HybridDecoderBlock(nn.Module):
    """Hybrid Decoder combining TCN and Transformer"""
    def __init__(self, d_model, output_dim, tcn_kernel_size=3, tcn_num_layers=2, 
                 transformer_nhead=8, transformer_num_layers=2, dim_feedforward=512, dropout=0.1):
        super(HybridDecoderBlock, self).__init__()
        
        # TCN branch for local features
        self.tcn_branch = TCNDecoderBlock(
            d_model=d_model,
            output_dim=d_model,  # Keep same dimension for combination
            kernel_size=tcn_kernel_size,
            num_layers=tcn_num_layers,
            dropout=dropout
        )
        
        # Transformer branch for global features
        self.transformer_branch = TransformerDecoderBlock(
            d_model=d_model,
            output_dim=d_model,  # Keep same dimension for combination
            nhead=transformer_nhead,
            num_layers=transformer_num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Feature combination
        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Output tensor of shape (batch_size, seq_len, output_dim)
        """
        # TCN branch for local temporal patterns
        tcn_output = self.tcn_branch(x)
        
        # Transformer branch for global dependencies
        transformer_output = self.transformer_branch(x)
        
        # Combine features
        combined = torch.cat([tcn_output, transformer_output], dim=-1)  # (batch_size, seq_len, d_model*2)
        fused = self.feature_fusion(combined)  # (batch_size, seq_len, d_model)
        
        # Final output projection
        output = self.output_projection(fused)
        
        return output


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
    """Decoder with multiple architecture options for reconstruction"""
    def __init__(self, 
                 d_model, 
                 output_dim,
                 decoder_type='mlp',
                 hidden_dims=None,
                 dropout=0.1,
                 # TCN-specific parameters
                 tcn_kernel_size=3,
                 tcn_num_layers=3,
                 # Transformer-specific parameters
                 transformer_nhead=8,
                 transformer_num_layers=3,
                 dim_feedforward=512,
                 # Hybrid-specific parameters
                 hybrid_tcn_kernel_size=3,
                 hybrid_tcn_num_layers=2,
                 hybrid_transformer_nhead=8,
                 hybrid_transformer_num_layers=2,
                 hybrid_dim_feedforward=512):
        """
        Args:
            d_model: Model dimension from encoder
            output_dim: Output dimension (should match input dimension of original data)
            decoder_type: Type of decoder ('mlp', 'tcn', 'transformer', 'hybrid', 'custom_linear')
            hidden_dims: List of hidden dimensions for MLP (default: [d_model, d_model//2, d_model//4])
            dropout: Dropout rate
            # TCN parameters
            tcn_kernel_size: Kernel size for TCN layers
            tcn_num_layers: Number of TCN layers
            # Transformer parameters
            transformer_nhead: Number of attention heads for transformer
            transformer_num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension for transformer
            # Hybrid parameters
            hybrid_tcn_kernel_size: Kernel size for TCN in hybrid decoder
            hybrid_tcn_num_layers: Number of TCN layers in hybrid decoder
            hybrid_transformer_nhead: Number of attention heads for transformer in hybrid decoder
            hybrid_transformer_num_layers: Number of transformer layers in hybrid decoder
            hybrid_dim_feedforward: Feedforward dimension for transformer in hybrid decoder
        """
        super(Decoder, self).__init__()
        
        self.decoder_type = decoder_type
        self.d_model = d_model
        self.output_dim = output_dim
        
        # Initialize decoder based on type
        if decoder_type == 'mlp':
            if hidden_dims is None:
                hidden_dims = [d_model, d_model // 2, d_model // 4]
            
            self.decoder_block = MLPDecoderBlock(
                d_model=d_model,
                output_dim=output_dim,
                hidden_dims=hidden_dims,
                dropout=dropout
            )
            
        elif decoder_type == 'tcn':
            self.decoder_block = TCNDecoderBlock(
                d_model=d_model,
                output_dim=output_dim,
                kernel_size=tcn_kernel_size,
                num_layers=tcn_num_layers,
                dropout=dropout
            )
            
        elif decoder_type == 'transformer':
            self.decoder_block = TransformerDecoderBlock(
                d_model=d_model,
                output_dim=output_dim,
                nhead=transformer_nhead,
                num_layers=transformer_num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            
        elif decoder_type == 'hybrid':
            self.decoder_block = HybridDecoderBlock(
                d_model=d_model,
                output_dim=output_dim,
                tcn_kernel_size=hybrid_tcn_kernel_size,
                tcn_num_layers=hybrid_tcn_num_layers,
                transformer_nhead=hybrid_transformer_nhead,
                transformer_num_layers=hybrid_transformer_num_layers,
                dim_feedforward=hybrid_dim_feedforward,
                dropout=dropout
            )
            
        elif decoder_type == 'custom_linear':
            self.decoder_block = CustomLinearDecoderBlock(
                d_model=d_model,
                output_dim=output_dim,
                dropout=dropout
            )
            
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}. "
                           f"Supported types: 'mlp', 'tcn', 'transformer', 'hybrid', 'custom_linear'")
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model) - encoded augmented data
        Returns:
            Reconstructed output tensor of shape (batch_size, seq_len, output_dim)
        """
        output = self.decoder_block(x)
        return output
    
    def get_individual_outputs(self, x):
        """
        Get outputs for analysis (for compatibility)
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Dictionary with decoder output
        """
        with torch.no_grad():
            decoder_output = self.decoder_block(x)
            
            return {
                self.decoder_type: decoder_output
            }
    
    def get_decoder_info(self):
        """Get information about the decoder architecture"""
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'decoder_type': self.decoder_type,
            'd_model': self.d_model,
            'output_dim': self.output_dim,
            'total_parameters': total_params
        }


# Example usage and testing
if __name__ == "__main__":
    # Test parameters
    batch_size = 32
    seq_len = 100
    d_model = 256
    output_dim = 128
    
    print("=" * 80)
    print("TESTING ALL DECODER TYPES")
    print("=" * 80)
    
    decoder_types = ['mlp', 'tcn', 'transformer', 'hybrid']
    
    for decoder_type in decoder_types:
        print(f"\n{'='*60}")
        print(f"Testing {decoder_type.upper()} Decoder:")
        print(f"{'='*60}")
        
        try:
            decoder = Decoder(
                d_model=d_model,
                output_dim=output_dim,
                decoder_type=decoder_type,
                hidden_dims=[d_model, d_model//2, d_model//4],
                dropout=0.1,
                # TCN parameters
                tcn_kernel_size=3,
                tcn_num_layers=2,
                # Transformer parameters
                transformer_nhead=8,
                transformer_num_layers=2,
                dim_feedforward=512,
                # Hybrid parameters
                hybrid_tcn_kernel_size=3,
                hybrid_tcn_num_layers=1,
                hybrid_transformer_nhead=4,
                hybrid_transformer_num_layers=1,
                hybrid_dim_feedforward=256
            )
            
            x = torch.randn(batch_size, seq_len, d_model)  # Encoded augmented data
            output = decoder(x)
            
            print(f"✅ Input shape: {x.shape}")
            print(f"✅ Output shape: {output.shape}")
            print(f"✅ Shape match: {x.shape[:-1] == output.shape[:-1] and output.shape[-1] == output_dim}")
            
            # Test individual outputs
            individual_outputs = decoder.get_individual_outputs(x)
            for name, output_tensor in individual_outputs.items():
                print(f"✅ Individual output '{name}': {output_tensor.shape}")
            
            # Get decoder info
            decoder_info = decoder.get_decoder_info()
            print(f"✅ Decoder type: {decoder_info['decoder_type']}")
            print(f"✅ Total parameters: {decoder_info['total_parameters']:,}")
            
            # Verify output dimensions
            expected_shape = (batch_size, seq_len, output_dim)
            if output.shape == expected_shape:
                print(f"✅ Shape verification passed: {output.shape} == {expected_shape}")
            else:
                print(f"❌ Shape verification failed: {output.shape} != {expected_shape}")
                
        except Exception as e:
            print(f"❌ Error testing {decoder_type} decoder: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("DECODER TESTING COMPLETED")
    print(f"{'='*80}")
