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
                 use_wandb: bool = True,
                 project_name: str = 'contrastive-learning',
                 experiment_name: str = None):
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
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.save_dir = save_dir
        self.use_wandb = use_wandb
        
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
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.contrastive_losses = []
        self.reconstruction_losses = []
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize wandb
        if self.use_wandb:
            if experiment_name is None:
                experiment_name = f"contrastive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            wandb.init(
                project=project_name,
                name=experiment_name,
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
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_contrastive_loss = 0.0
        total_reconstruction_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_dataloader, desc="Training")
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
        
        # Update learning rate
        self.scheduler.step()
        
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
    
    def train(self, num_epochs: int, save_every: int = 10) -> Dict[str, List[float]]:
        """Train the model for specified number of epochs"""
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Removed validation - only training
        
        # Create epoch progress bar
        epoch_pbar = tqdm(range(num_epochs), desc="Training Epochs", position=0)
        
        for epoch in epoch_pbar:
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # Training
            train_losses = self.train_epoch()
            self.train_losses.append(train_losses['total_loss'])
            self.contrastive_losses.append(train_losses['contrastive_loss'])
            self.reconstruction_losses.append(train_losses['reconstruction_loss'])
            
            print(f"Train Loss: {train_losses['total_loss']:.4f}")
            print(f"Train Contrastive Loss: {train_losses['contrastive_loss']:.4f}")
            print(f"Train Reconstruction Loss: {train_losses['reconstruction_loss']:.4f}")
            
            # Save checkpoint after every epoch
            self.save_checkpoint(epoch)
            
            # Save additional checkpoint every save_every epochs
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(epoch, suffix=f"_every_{save_every}")
            
            # Print learning rate
            current_lr = self.scheduler.get_last_lr()[0]
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'Train Loss': f"{train_losses['total_loss']:.4f}",
                'Contrastive': f"{train_losses['contrastive_loss']:.4f}",
                'Reconstruction': f"{train_losses['reconstruction_loss']:.4f}",
                'LR': f"{current_lr:.6f}"
            })
            
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
    
    def save_checkpoint(self, epoch: int, is_best: bool = False, suffix: str = ""):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'contrastive_losses': self.contrastive_losses,
            'reconstruction_losses': self.reconstruction_losses,
            'contrastive_weight': self.contrastive_weight,
            'reconstruction_weight': self.reconstruction_weight
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch + 1}{suffix}.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.contrastive_losses = checkpoint.get('contrastive_losses', [])
        self.reconstruction_losses = checkpoint.get('reconstruction_losses', [])
        
        print(f"Checkpoint loaded from {checkpoint_path}")
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
                                 window_size: int = 100,
                                 batch_size: int = 32,
                                 num_workers: int = 4,
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
    dataloaders = create_dataloaders(
        dataset_type=dataset_type,
        data_path=data_path,
        window_size=window_size,
        stride=window_size,  # Non-overlapping windows
        batch_size=batch_size,
        num_workers=num_workers,
        **kwargs
    )
    
    # Get training data
    train_dataset = dataloaders['train'].dataset
    
    # Create contrastive dataset
    contrastive_train_dataset = ContrastiveDataset(
        data=train_dataset.data,
        window_size=window_size,
        stride=window_size
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        contrastive_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Validation dataloader (optional)
    val_dataloader = None
    if 'val' in dataloaders:
        val_dataset = dataloaders['val'].dataset
        contrastive_val_dataset = ContrastiveDataset(
            data=val_dataset.data,
            window_size=window_size,
            stride=window_size
        )
        
        val_dataloader = DataLoader(
            contrastive_val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_dataloader, val_dataloader


# Example usage
if __name__ == "__main__":
    # Parameters
    dataset_type = 'ecg'
    data_path = 'datasets/ecg'
    window_size = 100
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
    history = trainer.train(num_epochs=num_epochs, save_every=10)
    
    # Plot training history
    trainer.plot_training_history('training_history.png')
    
    print("Training completed!")
