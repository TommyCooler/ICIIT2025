#!/usr/bin/env python3
"""
Main script for contrastive learning model training
"""

import argparse
import torch
import numpy as np
import os
import sys
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.contrastive_model import ContrastiveModel
from model.train_contrastive import ContrastiveTrainer, create_contrastive_dataloaders
from utils.dataloader import create_dataloaders


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Contrastive Learning for Time Series Anomaly Detection')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='ecg', 
                       choices=['ecg', 'psm', 'nab', 'smap_msl', 'smd'],
                       help='Type of dataset to use')
    parser.add_argument('--data_path', type=str, default='D:/Hoc_voi_cha_hanh/FPT/Hoc_rieng/ICIIT2025/MainModel/datasets/ecg',
                       help='Path to dataset directory')
    parser.add_argument('--dataset_name', type=str, default=None,
                       help='Specific dataset name (for nab, smap_msl, smd)')
    
    # Model arguments
    parser.add_argument('--input_dim', type=int, default=3,
                       help='Input dimension (number of features)')
    parser.add_argument('--d_model', type=int, default=256,
                       help='Model dimension for transformer')
    parser.add_argument('--projection_dim', type=int, default=128,
                       help='Dimension for contrastive learning projection')
    parser.add_argument('--nhead', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--transformer_layers', type=int, default=6,
                       help='Number of transformer encoder layers')
    parser.add_argument('--tcn_output_dim', type=int, default=None,
                       help='Output dimension for TCN')
    parser.add_argument('--tcn_kernel_size', type=int, default=3,
                       help='Kernel size for TCN')
    parser.add_argument('--tcn_num_layers', type=int, default=3,
                       help='Number of TCN layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--temperature', type=float, default=0.07,
                       help='Temperature for InfoNCE loss')
    parser.add_argument('--combination_method', type=str, default='concat',
                       choices=['concat', 'stack'],
                       help='Method for combining TCN and Transformer outputs')
    
    # Training arguments
    parser.add_argument('--window_size', type=int, default=100,
                       help='Size of windows')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--contrastive_weight', type=float, default=1.0,
                       help='Weight for contrastive loss')
    parser.add_argument('--reconstruction_weight', type=float, default=1.0,
                       help='Weight for reconstruction loss')
    parser.add_argument('--epsilon', type=float, default=1e-5,
                       help='Small constant for numerical stability in contrastive loss')
    
    # Wandb arguments
    parser.add_argument('--use_wandb', action='store_true', default=True,
                        help='Use wandb for logging')
    parser.add_argument('--no_wandb', dest='use_wandb', action='store_false',
                        help='Disable wandb logging')
    parser.add_argument('--project_name', type=str, default='contrastive-learning',
                        help='Wandb project name')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Wandb experiment name')
# Augmentation is handled by the model, not in dataloader
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of worker processes for data loading')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_arg: str) -> str:
    """Get device string"""
    if device_arg == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device_arg


def get_input_dim(dataset: str, data_path: str = None) -> int:
    """Get input dimension based on dataset type or by loading data"""
    # Try to load data to get actual input dimension
    if data_path and dataset == 'ecg':
        try:
            from src.utils.dataloader import DatasetFactory
            loader = DatasetFactory.create_loader(dataset, data_path, normalize=True)
            data = loader.load_all_datasets()
            input_dim = data['train_data'].shape[0]  # Number of features
            print(f"Detected input dimension for {dataset}: {input_dim}")
            return input_dim
        except Exception as e:
            print(f"Could not detect input dimension for {dataset}: {e}")
            print("Using default dimension")
    
    # Fallback to default dimensions
    dims = {
        'ecg': 2,  # Default fallback
        'psm': 25,
        'nab': 1,
        'smap_msl': 25,
        'smd': 38
    }
    return dims.get(dataset, 2)


def main():
    """Main function"""
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Get device
    device = get_device(args.device)
    
    # Get input dimension
    if args.input_dim is None:
        args.input_dim = get_input_dim(args.dataset, args.data_path)
    
    # Print configuration
    print("=" * 60)
    print("Contrastive Learning for Time Series Anomaly Detection")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Data path: {args.data_path}")
    print(f"Input dimension: {args.input_dim}")
    print(f"Window size: {args.window_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Device: {device}")
    print(f"Random seed: {args.seed}")
    print("=" * 60)
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"{args.dataset}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(save_dir, 'config.json')
    with open(config_path, 'w') as f:
        import json
        json.dump(vars(args), f, indent=2)
    print(f"Configuration saved to {config_path}")
    
    try:
        # Create dataloaders
        print("\nCreating dataloaders...")
        train_dataloader, val_dataloader = create_contrastive_dataloaders(
            dataset_type=args.dataset,
            data_path=args.data_path,
            dataset_name=args.dataset_name,
            window_size=args.window_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
# Augmentation is handled by the model
        )
        
        print(f"Train batches: {len(train_dataloader)}")
        if val_dataloader:
            print(f"Validation batches: {len(val_dataloader)}")
        
        # Create model
        print("\nCreating model...")
        model = ContrastiveModel(
            input_dim=args.input_dim,
            d_model=args.d_model,
            projection_dim=args.projection_dim,
            nhead=args.nhead,
            transformer_layers=args.transformer_layers,
            tcn_output_dim=args.tcn_output_dim,
            tcn_kernel_size=args.tcn_kernel_size,
            tcn_num_layers=args.tcn_num_layers,
            dropout=args.dropout,
            temperature=args.temperature,
            combination_method=args.combination_method
        )
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create trainer
        print("\nCreating trainer...")
        trainer = ContrastiveTrainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            contrastive_weight=args.contrastive_weight,
            reconstruction_weight=args.reconstruction_weight,
            epsilon=args.epsilon,
            device=device,
            save_dir=save_dir,
            use_wandb=args.use_wandb,
            project_name=args.project_name,
            experiment_name=args.experiment_name
        )
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if args.resume:
            print(f"\nResuming from checkpoint: {args.resume}")
            start_epoch = trainer.load_checkpoint(args.resume)
        
        # Train model
        print(f"\nStarting training from epoch {start_epoch + 1}...")
        trainer.train(
            num_epochs=args.num_epochs,
            save_every=args.save_every
        )
        
        # Plot training history
        plot_path = os.path.join(save_dir, 'training_history.png')
        trainer.plot_training_history(plot_path)
        
        # Save final model
        final_checkpoint_path = os.path.join(save_dir, 'final_model.pt')
        trainer.save_checkpoint(args.num_epochs - 1)
        print(f"\nFinal model saved to {final_checkpoint_path}")
        
        print("\nTraining completed successfully!")
        print(f"Results saved to: {save_dir}")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
