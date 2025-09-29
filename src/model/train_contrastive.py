import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import wandb

from .contrastive_model import ContrastiveModel, ContrastiveDataset
from utils.dataloader import create_dataloaders


class ContrastiveTrainer:
    """Trainer for contrastive learning model"""
    
    def __init__(self,
                 model: ContrastiveModel,
                 train_dataloader: DataLoader,
                 val_dataloader: Optional[DataLoader] = None,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 contrastive_weight: float = 1.0,
                 reconstruction_weight: float = 1.0,
                 epsilon: float = 1e-5,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 save_dir: str = 'checkpoints',
                 use_wandb: bool = False,
                 project_name: str = 'contrastive-learning',
                 experiment_name: str = None,
                 use_lr_scheduler: bool = False,
                 scheduler_type: str = 'cosine',
                 scheduler_params: Optional[Dict] = None,
                 window_size: int = None):
        """
        Args:
            model: Contrastive learning model
            train_dataloader: Training dataloader
            val_dataloader: Validation dataloader (optional)
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            contrastive_weight: Weight for contrastive loss
            reconstruction_weight: Weight for reconstruction loss
            epsilon: Small constant for numerical stability in contrastive loss
            device: Device to run training on
            save_dir: Directory to save checkpoints
            use_wandb: Whether to use wandb for logging
            project_name: Wandb project name
            experiment_name: Wandb experiment name
            use_lr_scheduler: Whether to use learning rate scheduler
            scheduler_type: Type of scheduler ('cosine', 'step', 'exponential', 'plateau')
            scheduler_params: Additional parameters for scheduler
        """
        # Force CUDA usage for training
        self.model = model.to('cuda')
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = 'cuda'
        self.save_dir = save_dir
        self.use_wandb = use_wandb
        self.window_size = window_size
        
        # Loss weights
        self.contrastive_weight = contrastive_weight
        self.reconstruction_weight = reconstruction_weight
        self.epsilon = epsilon
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler setup
        self.use_lr_scheduler = use_lr_scheduler
        self.scheduler_type = scheduler_type
        self.scheduler_params = scheduler_params or {}
        self.scheduler = None
        
        if self.use_lr_scheduler:
            self._setup_scheduler()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.contrastive_losses = []
        self.reconstruction_losses = []
        
        # Best loss tracking for checkpoint saving
        self.best_loss = float('inf')
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize wandb
        if self.use_wandb:
            if experiment_name is None:
                experiment_name = f"contrastive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            wandb.init(
                project=project_name,
                name=experiment_name,
                resume="allow",
                config={
                    'learning_rate': learning_rate,
                    'weight_decay': weight_decay,
                    'contrastive_weight': contrastive_weight,
                    'reconstruction_weight': reconstruction_weight,
                    'epsilon': epsilon,
                    'device': device,
                    'model_params': sum(p.numel() for p in model.parameters()),
                    'train_batches': len(train_dataloader),
                    'val_batches': len(val_dataloader) if val_dataloader else 0
                }
            )
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler based on type"""
        if self.scheduler_type == 'cosine':
            # Cosine annealing scheduler
            t_max = self.scheduler_params.get('T_max', 100)
            eta_min = self.scheduler_params.get('eta_min', 1e-6)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=t_max,
                eta_min=eta_min
            )
            
        elif self.scheduler_type == 'step':
            # Step scheduler
            step_size = self.scheduler_params.get('step_size', 30)
            gamma = self.scheduler_params.get('gamma', 0.1)
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
            
        elif self.scheduler_type == 'exponential':
            # Exponential scheduler
            gamma = self.scheduler_params.get('gamma', 0.95)
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=gamma
            )
            
        elif self.scheduler_type == 'plateau':
            # Reduce on plateau scheduler
            mode = self.scheduler_params.get('mode', 'min')
            factor = self.scheduler_params.get('factor', 0.5)
            patience = self.scheduler_params.get('patience', 10)
            threshold = self.scheduler_params.get('threshold', 1e-4)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=mode,
                factor=factor,
                patience=patience,
                threshold=threshold
            )
            
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")
    
    def get_current_lr(self) -> float:
        """Get current learning rate"""
        if self.scheduler is not None:
            return self.scheduler.get_last_lr()[0]
        else:
            return self.optimizer.param_groups[0]['lr']
    
    def update_lr_scheduler(self, metric: Optional[float] = None):
        """Update learning rate scheduler"""
        if self.scheduler is not None:
            if self.scheduler_type == 'plateau' and metric is not None:
                self.scheduler.step(metric)
            else:
                self.scheduler.step()
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_contrastive_loss = 0.0
        total_reconstruction_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(
            self.train_dataloader,
            desc="Training",
            position=1,
            leave=False,
            dynamic_ncols=True
        )
        for original_batch, augmented_batch in pbar:
            # Move to device
            original_batch = original_batch.to(self.device)
            augmented_batch = augmented_batch.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            self.model(original_batch, augmented_batch)
            
            # Compute losses
            losses = self.model.compute_total_loss(
                original_batch,
                augmented_batch,
                contrastive_weight=self.contrastive_weight,
                reconstruction_weight=self.reconstruction_weight,
                epsilon=self.epsilon
            )
            
            # Backward pass
            losses['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update parameters
            self.optimizer.step()
            
            # Update metrics
            total_loss += losses['total_loss'].item()
            total_contrastive_loss += losses['contrastive_loss'].item()
            total_reconstruction_loss += losses['reconstruction_loss'].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{losses['total_loss'].item():.4f}",
                'Contrastive': f"{losses['contrastive_loss'].item():.4f}",
                'Reconstruction': f"{losses['reconstruction_loss'].item():.4f}"
            })
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'batch/total_loss': losses['total_loss'].item(),
                    'batch/contrastive_loss': losses['contrastive_loss'].item(),
                    'batch/reconstruction_loss': losses['reconstruction_loss'].item(),
                    'batch/learning_rate': self.optimizer.param_groups[0]['lr']
                })
        
        # Update learning rate scheduler
        self.update_lr_scheduler()
        
        # Compute average losses
        avg_losses = {
            'total_loss': total_loss / num_batches,
            'contrastive_loss': total_contrastive_loss / num_batches,
            'reconstruction_loss': total_reconstruction_loss / num_batches
        }
        
        # Log epoch-level metrics to wandb
        if self.use_wandb:
            wandb.log({
                'epoch/train_total_loss': avg_losses['total_loss'],
                'epoch/train_contrastive_loss': avg_losses['contrastive_loss'],
                'epoch/train_reconstruction_loss': avg_losses['reconstruction_loss'],
                'epoch/learning_rate': self.optimizer.param_groups[0]['lr']
            })
        
        return avg_losses
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        total_contrastive_loss = 0.0
        total_reconstruction_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_dataloader, desc="Validation")
            for original_batch, augmented_batch in pbar:
                # Move to device
                original_batch = original_batch.to(self.device)
                augmented_batch = augmented_batch.to(self.device)
                
                # Forward pass
                self.model(original_batch, augmented_batch)
                
                # Compute losses
                losses = self.model.compute_total_loss(
                    original_batch,
                    augmented_batch,
                    contrastive_weight=self.contrastive_weight,
                    reconstruction_weight=self.reconstruction_weight,
                    epsilon=self.epsilon
                )
                
                # Update metrics
                total_loss += losses['total_loss'].item()
                total_contrastive_loss += losses['contrastive_loss'].item()
                total_reconstruction_loss += losses['reconstruction_loss'].item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f"{losses['total_loss'].item():.4f}",
                    'Contrastive': f"{losses['contrastive_loss'].item():.4f}",
                    'Reconstruction': f"{losses['reconstruction_loss'].item():.4f}"
                })
                
                # Log to wandb
                if self.use_wandb:
                    wandb.log({
                        'val_batch/total_loss': losses['total_loss'].item(),
                        'val_batch/contrastive_loss': losses['contrastive_loss'].item(),
                        'val_batch/reconstruction_loss': losses['reconstruction_loss'].item()
                    })
        
        # Compute average losses
        avg_losses = {
            'total_loss': total_loss / num_batches,
            'contrastive_loss': total_contrastive_loss / num_batches,
            'reconstruction_loss': total_reconstruction_loss / num_batches
        }
        
        # Log validation epoch-level metrics to wandb
        if self.use_wandb:
            wandb.log({
                'epoch/val_total_loss': avg_losses['total_loss'],
                'epoch/val_contrastive_loss': avg_losses['contrastive_loss'],
                'epoch/val_reconstruction_loss': avg_losses['reconstruction_loss']
            })
        
        return avg_losses
    
    def train(self, num_epochs: int, start_epoch: int = 0) -> Dict[str, List[float]]:
        """Train the model for specified number of epochs"""
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Removed validation - only training
        
        # Epoch loop with tqdm progress bar
        epoch_indices = range(start_epoch, start_epoch + num_epochs)
        with tqdm(total=num_epochs, desc="Epochs", position=0, leave=True, dynamic_ncols=True) as epoch_bar:
            for e_idx, epoch in enumerate(epoch_indices, start=1):
                epoch_bar.set_postfix({ 'epoch': f"{e_idx}/{num_epochs}" })
                print(f"\nEpoch {epoch + 1}")
                print("-" * 50)
            
            # Training
            train_losses = self.train_epoch()
            self.train_losses.append(train_losses['total_loss'])
            self.contrastive_losses.append(train_losses['contrastive_loss'])
            self.reconstruction_losses.append(train_losses['reconstruction_loss'])
            
            print(f"Train Loss: {train_losses['total_loss']:.4f}")
            print(f"Train Contrastive Loss: {train_losses['contrastive_loss']:.4f}")
            print(f"Train Reconstruction Loss: {train_losses['reconstruction_loss']:.4f}")
            
            # Save checkpoint only if loss is better than previous best
            self.save_checkpoint(epoch, current_loss=train_losses['total_loss'])
            
            # Print learning rate
            current_lr = self.get_current_lr()
            print(f"Learning Rate: {current_lr:.6f}")
            # Reflect key metrics on the epoch bar
            epoch_bar.set_postfix({
                'epoch': f"{e_idx}/{num_epochs}",
                'loss': f"{train_losses['total_loss']:.4f}",
                'ctr': f"{train_losses['contrastive_loss']:.4f}",
                'rec': f"{train_losses['reconstruction_loss']:.4f}"
            })
            
            # Update epoch progress bar
            epoch_bar.update(1)
            
            # Log epoch summary to wandb
            if self.use_wandb:
                log_dict = {
                    'epoch/epoch': epoch + 1,
                    'epoch/learning_rate': current_lr,
                    'epoch/train_total_loss': train_losses['total_loss'],
                    'epoch/train_contrastive_loss': train_losses['contrastive_loss'],
                    'epoch/train_reconstruction_loss': train_losses['reconstruction_loss']
                }
                wandb.log(log_dict)
        
        print("\nTraining completed!")
        
        # Finish wandb run
        if self.use_wandb:
            wandb.finish()
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'contrastive_losses': self.contrastive_losses,
            'reconstruction_losses': self.reconstruction_losses
        }
    
    def save_checkpoint(self, epoch: int, current_loss: float = None):
        """Save model checkpoint only if loss is better than previous best"""
        # If current_loss is provided, check if it's better than best_loss
        should_save = False
        if current_loss is not None:
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                should_save = True
                print(f"New best loss: {current_loss:.4f} (previous: {self.best_loss:.4f})")
            else:
                print(f"Loss {current_loss:.4f} not better than best {self.best_loss:.4f}, skipping checkpoint save")
        else:
            # If no current_loss provided, save anyway (for final model)
            should_save = True
        
        if not should_save:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'contrastive_losses': self.contrastive_losses,
            'reconstruction_losses': self.reconstruction_losses,
            'contrastive_weight': self.contrastive_weight,
            'reconstruction_weight': self.reconstruction_weight,
            'best_loss': self.best_loss,
            # Add window_size for inference compatibility
            'window_size': getattr(self, 'window_size', None)
        }
        
        # Save checkpoint with fixed filename (overwrite previous)
        checkpoint_path = os.path.join(self.save_dir, 'best_model.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Best checkpoint saved: {checkpoint_path} (loss: {self.best_loss:.4f})")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler if available
        if checkpoint.get('scheduler_state_dict') is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.contrastive_losses = checkpoint.get('contrastive_losses', [])
        self.reconstruction_losses = checkpoint.get('reconstruction_losses', [])
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Best loss from checkpoint: {self.best_loss:.4f}")
        return checkpoint['epoch']
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        _, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total loss
        axes[0, 0].plot(self.train_losses, label='Train', color='blue')
        if self.val_losses:
            axes[0, 0].plot(self.val_losses, label='Validation', color='red')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Contrastive loss
        axes[0, 1].plot(self.contrastive_losses, label='Train', color='green')
        axes[0, 1].set_title('Contrastive Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Reconstruction loss
        axes[1, 0].plot(self.reconstruction_losses, label='Train', color='orange')
        axes[1, 0].set_title('Reconstruction Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate
        if self.scheduler is not None:
            LEARNING_RATE_LABEL = 'Learning Rate'
            lr_history = [self.scheduler.get_last_lr()[0] for _ in range(len(self.train_losses))]
            axes[1, 1].plot(lr_history, label=LEARNING_RATE_LABEL, color='purple')
            axes[1, 1].set_title(LEARNING_RATE_LABEL)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel(LEARNING_RATE_LABEL)
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()


def create_contrastive_dataloaders(dataset_type: str,
                                 data_path: str,
                                 window_size: int = 128,
                                 batch_size: int = 32,
                                 num_workers: int = 4,
                                 mask_mode: str = 'none',
                                 mask_ratio: float = 0.0,
                                 mask_seed: int = None,
                                 **kwargs) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create dataloaders for contrastive learning
    
    Args:
        dataset_type: Type of dataset
        data_path: Path to dataset
        window_size: Size of windows
        batch_size: Batch size
        num_workers: Number of workers
        **kwargs: Additional arguments for dataset loading
    
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Load dataset using existing dataloader
    # Disable validation split/use for training pipeline
    dataloaders = create_dataloaders(
        dataset_type=dataset_type,
        data_path=data_path,
        window_size=window_size,
        stride=1,  # Non-overlapping windows
        batch_size=batch_size,
        num_workers=num_workers,
        validation_ratio=0.0,
        **kwargs
    )
    
    # Get training data
    train_dataset = dataloaders['train'].dataset
    
    # Create contrastive dataset
    contrastive_train_dataset = ContrastiveDataset(
        data=train_dataset.data,
        window_size=window_size,
        stride=1,
        mask_mode=mask_mode,
        mask_ratio=mask_ratio,
        mask_seed=mask_seed
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        contrastive_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Do not construct validation dataloader (not used)
    val_dataloader = None
    
    return train_dataloader, val_dataloader


# Example usage
if __name__ == "__main__":
    # Parameters
    dataset_type = 'ecg'
    data_path = 'datasets/ecg'
    window_size = 128
    batch_size = 32
    num_epochs = 50
    
    # Create dataloaders
    train_dataloader, val_dataloader = create_contrastive_dataloaders(
        dataset_type=dataset_type,
        data_path=data_path,
        window_size=window_size,
        batch_size=batch_size
    )
    
    # Create model
    model = ContrastiveModel(
        input_dim=2,  # ECG has 2 channels
        d_model=256,
        projection_dim=128,
        nhead=8,
        transformer_layers=6,
        dropout=0.1
    )
    
    # Create trainer
    trainer = ContrastiveTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=1e-4,
        contrastive_weight=1.0,
        reconstruction_weight=1.0,
        save_dir='checkpoints/contrastive'
    )
    
    # Train model
    history = trainer.train(num_epochs=num_epochs)
    
    # Plot training history
    trainer.plot_training_history('training_history.png')
    
    print("Training completed!")
