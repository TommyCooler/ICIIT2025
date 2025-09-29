#!/usr/bin/env python3
"""
Main script for contrastive learning model training
"""

import argparse
import json
from typing import List, Optional
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
                       choices=['ecg', 'psm', 'nab', 'smap_msl', 'smd', 'ucr'],
                       help='Type of dataset to use')
    parser.add_argument('--data_path', type=str, default='D:\Hoc_voi_cha_hanh\FPT\Hoc_rieng\ICIIT2025\MainModel\datasets\ecg',
                       help='Path to dataset directory')
    parser.add_argument('--dataset_name', type=str, default='chfdb_chf01_275.pkl',
                       help='Specific dataset name (for ecg, nab, smap_msl, smd)')
    
    # Model arguments
    parser.add_argument('--input_dim', type=int, default=None,
                       help='Input dimension (number of features). If not set, auto-detected')
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
    parser.add_argument('--tcn_num_layers', type=int, default=4,
                       help='Number of TCN layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--temperature', type=float, default=1,
                       help='Temperature for InfoNCE loss')
    parser.add_argument('--combination_method', type=str, default='stack',
                       choices=['concat', 'stack'],
                       help='Method for combining TCN and Transformer outputs')

    # Augmentation-specific overrides (distinct names to avoid confusion with encoder)
    parser.add_argument('--aug_nhead', type=int, default=2,
                       help='Augmentation transformer nhead (override; default: model nhead)')
    parser.add_argument('--aug_num_layers', type=int, default=1,
                       help='Augmentation transformer number of layers')
    parser.add_argument('--aug_tcn_kernel_size', type=int, default=3,
                       help='Augmentation TCN kernel size (override; default: model tcn_kernel_size)')
    parser.add_argument('--aug_tcn_num_layers', type=int, default=1,
                       help='Augmentation TCN number of layers')
    parser.add_argument('--aug_dropout', type=float, default=0.1,
                       help='Augmentation dropout (override; default: model dropout)')
    parser.add_argument('--aug_temperature', type=float, default=None,
                       help='Augmentation temperature (override; default: model temperature)')
    parser.add_argument('--use_contrastive', action='store_true', default=False,
                       help='Use contrastive learning branch')
    parser.add_argument('--no_contrastive', dest='use_contrastive', action='store_false',
                       help='Disable contrastive learning branch')
    
    # Training arguments
    parser.add_argument('--window_size', type=int, default=128,
                       help='Size of windows')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10,
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
    # Masking options for training (augmented input masking)
    parser.add_argument('--mask_mode', type=str, default='time', choices=['none', 'time', 'feature'],
                       help='Masking mode for augmented input during training')
    parser.add_argument('--mask_ratio', type=float, default=0.2,
                       help='Fraction of timesteps/features to mask (0.0 - 1.0)')
    parser.add_argument('--mask_seed', type=int, default=None,
                       help='Random seed for masking reproducibility')
    
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
    
    # LR scheduler arguments
    parser.add_argument('--use_lr_scheduler', action='store_true', default=True,
                       help='Use learning rate scheduler')
    parser.add_argument('--scheduler_type', type=str, default='cosine',
                       choices=['cosine', 'step', 'exponential', 'plateau'],
                       help='Learning rate scheduler type')
    parser.add_argument('--scheduler_params', type=str, default='{}',
                       help='JSON string of scheduler params, e.g. {"T_max": 100, "eta_min": 1e-6}')

    # System arguments
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of worker processes for data loading')
    parser.add_argument('--save_dir', type=str, default=r'checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default="latest",
                       help='Path to checkpoint to resume from, or "latest" to resume from latest checkpoint')
    parser.add_argument('--resume_epoch', type=int, default=None,
                       help='Specific epoch to resume from (if not specified, resume from last epoch in checkpoint)')
    parser.add_argument('--list_checkpoints', action='store_true',
                       help='List available checkpoints in save directory')
    
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


def list_checkpoints(save_dir: str) -> List[str]:
    """List available checkpoints in save directory"""
    if not os.path.exists(save_dir):
        print(f"Save directory {save_dir} does not exist")
        return []
    
    checkpoint_files = []
    for file in os.listdir(save_dir):
        if file.endswith('.pth') and 'checkpoint_epoch_' in file:
            checkpoint_files.append(os.path.join(save_dir, file))
    
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    print(f"\nAvailable checkpoints in {save_dir}:")
    for i, checkpoint in enumerate(checkpoint_files):
        filename = os.path.basename(checkpoint)
        epoch = filename.split('_')[-1].split('.')[0]
        print(f"  {i+1}. {filename} (Epoch {epoch})")
    
    return checkpoint_files


def find_latest_checkpoint(save_dir: str) -> Optional[str]:
    """Find the latest checkpoint in save directory"""
    checkpoints = list_checkpoints(save_dir)
    if not checkpoints:
        return None
    return checkpoints[-1]


def get_input_dim(dataset: str, data_path: str = None) -> int:
    """Get input dimension based on dataset type or by loading data"""
    # For ECG dataset, always use 2 features (extracted from 3-column format)
    if dataset == 'ecg':
        print("ECG dataset: Using input_dim=2 (2 features extracted from [feature1, feature2, label] format)")
        return 2
    
    # Try to load data to get actual input dimension for other datasets
    if data_path:
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
        'ecg': 2,   # ECG standardized to 2 features
        'psm': 25,
        'nab': 1,
        'smap_msl': 25,
        'smd': 38,
        'ucr': 1    # UCR labeled arrays are (time,) so feature dim is 1
    }
    return dims.get(dataset, 2)


def main():
    """Main function"""
    args = parse_args()
    
    # Handle list checkpoints
    if args.list_checkpoints:
        list_checkpoints(args.save_dir)
        return 0
    
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
    print(f"Use contrastive: {args.use_contrastive}")
    print(f"Use wandb: {args.use_wandb}")
    print(f"Device: {device}")
    print(f"Random seed: {args.seed}")
    print("=" * 60)
    
    # Create save directory with timestamp to distinguish different training runs
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"{args.dataset}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save directory: {save_dir}")
    
    # Save configuration
    config_path = os.path.join(save_dir, 'config.json')
    with open(config_path, 'w') as f:
        import json
        json.dump(vars(args), f, indent=2)
    print(f"Configuration saved to {config_path}")
    
    try:
        # Create dataloaders
        print("\nCreating dataloaders (no validation)...")
        train_dataloader, val_dataloader = create_contrastive_dataloaders(
            dataset_type=args.dataset,
            data_path=args.data_path,
            dataset_name=args.dataset_name,
            window_size=args.window_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            mask_mode=args.mask_mode,
            mask_ratio=max(0.0, min(1.0, float(args.mask_ratio))),
            mask_seed=args.mask_seed
        )
        
        print(f"Train batches: {len(train_dataloader)}")
        # Validation is disabled; do not print val batches

        # Detect feature dimension from a sample batch to ensure consistency
        try:
            sample_orig, _ = next(iter(train_dataloader))
            detected_input_dim = int(sample_orig.shape[-1])
            if args.input_dim is None or args.input_dim != detected_input_dim:
                print(f"Detected input_dim from data: {detected_input_dim} (overriding args.input_dim={args.input_dim})")
                args.input_dim = detected_input_dim
        except Exception as e:
            print(f"Warning: Could not detect input_dim from dataloader: {e}")
        
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
            combination_method=args.combination_method,
            use_contrastive=args.use_contrastive,
            augmentation_kwargs={
                # Only pass if provided; ContrastiveModel will fallback to model params
                **({ 'nhead': args.aug_nhead } if args.aug_nhead is not None else {}),
                **({ 'tcn_kernel_size': args.aug_tcn_kernel_size } if args.aug_tcn_kernel_size is not None else {}),
                **({ 'tcn_num_layers': args.aug_tcn_num_layers } if args.aug_tcn_num_layers is not None else {}),
                'num_layers': args.aug_num_layers,
                **({ 'dropout': args.aug_dropout } if args.aug_dropout is not None else {}),
                **({ 'temperature': args.aug_temperature } if args.aug_temperature is not None else {}),
            }
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
            experiment_name=args.experiment_name,
            window_size=args.window_size,
            use_lr_scheduler=args.use_lr_scheduler,
            scheduler_type=args.scheduler_type,
            scheduler_params=(json.loads(args.scheduler_params) if isinstance(args.scheduler_params, str) and args.scheduler_params else {})
        )
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if args.resume:
            if args.resume == 'latest':
                # Find latest checkpoint
                latest_checkpoint = find_latest_checkpoint(save_dir)
                if latest_checkpoint:
                    print(f"\nResuming from latest checkpoint: {latest_checkpoint}")
                    start_epoch = trainer.load_checkpoint(latest_checkpoint)
                else:
                    print(f"\nNo checkpoints found in {save_dir}, starting from scratch")
            else:
                print(f"\nResuming from checkpoint: {args.resume}")
                start_epoch = trainer.load_checkpoint(args.resume)
            
            # Override start epoch if specified
            if args.resume_epoch is not None:
                print(f"Note: resume_epoch specified ({args.resume_epoch}) but will continue from checkpoint epoch ({start_epoch})")
        
        # Train model
        print(f"\nStarting training from epoch {start_epoch + 1}...")
        trainer.train(
            num_epochs=args.num_epochs,
            start_epoch=start_epoch
        )
        
        # Plot training history
        plot_path = os.path.join(save_dir, 'training_history.png')
        trainer.plot_training_history(plot_path)
        
        # Save final model (force save regardless of loss)
        final_checkpoint_path = os.path.join(save_dir, 'final_model.pth')
        trainer.save_checkpoint(args.num_epochs - 1, current_loss=None)
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
