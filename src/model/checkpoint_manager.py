#!/usr/bin/env python3
"""
Checkpoint Manager for Contrastive Learning Model
Provides utilities to manage, list, and resume from checkpoints
"""

import os
import torch
import argparse
from typing import List, Dict, Optional
from datetime import datetime


class CheckpointManager:
    """Manager for model checkpoints"""
    
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def list_checkpoints(self) -> List[Dict]:
        """List all available checkpoints with metadata"""
        checkpoints = []
        
        if not os.path.exists(self.save_dir):
            return checkpoints
        
        for file in os.listdir(self.save_dir):
            if file.endswith('.pth') and 'checkpoint_epoch_' in file:
                filepath = os.path.join(self.save_dir, file)
                try:
                    # Load checkpoint to get metadata
                    checkpoint = torch.load(filepath, map_location='cpu')
                    
                    checkpoint_info = {
                        'filename': file,
                        'filepath': filepath,
                        'epoch': checkpoint.get('epoch', 0),
                        'size_mb': os.path.getsize(filepath) / (1024 * 1024),
                        'modified_time': datetime.fromtimestamp(os.path.getmtime(filepath)),
                        'train_losses': len(checkpoint.get('train_losses', [])),
                        'has_scheduler': checkpoint.get('scheduler_state_dict') is not None
                    }
                    checkpoints.append(checkpoint_info)
                except Exception as e:
                    print(f"Warning: Could not load metadata for {file}: {e}")
        
        # Sort by epoch
        checkpoints.sort(key=lambda x: x['epoch'])
        return checkpoints
    
    def print_checkpoints(self):
        """Print formatted list of checkpoints"""
        checkpoints = self.list_checkpoints()
        
        if not checkpoints:
            print(f"No checkpoints found in {self.save_dir}")
            return
        
        print(f"\nAvailable checkpoints in {self.save_dir}:")
        print("-" * 80)
        print(f"{'#':<3} {'Epoch':<6} {'Size (MB)':<10} {'Modified':<20} {'Losses':<8} {'Scheduler':<10}")
        print("-" * 80)
        
        for i, cp in enumerate(checkpoints):
            print(f"{i+1:<3} {cp['epoch']+1:<6} {cp['size_mb']:<10.1f} "
                  f"{cp['modified_time'].strftime('%Y-%m-%d %H:%M'):<20} "
                  f"{cp['train_losses']:<8} {'Yes' if cp['has_scheduler'] else 'No':<10}")
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint"""
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
        return checkpoints[-1]['filepath']
    
    def get_checkpoint_by_epoch(self, epoch: int) -> Optional[str]:
        """Get path to checkpoint by epoch number"""
        checkpoints = self.list_checkpoints()
        for cp in checkpoints:
            if cp['epoch'] == epoch:
                return cp['filepath']
        return None
    
    def clean_old_checkpoints(self, keep_last: int = 5):
        """Keep only the last N checkpoints, delete others"""
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= keep_last:
            print(f"Only {len(checkpoints)} checkpoints found, keeping all")
            return
        
        # Keep the last N checkpoints
        to_delete = checkpoints[:-keep_last]
        
        print(f"Deleting {len(to_delete)} old checkpoints...")
        for cp in to_delete:
            try:
                os.remove(cp['filepath'])
                print(f"Deleted: {cp['filename']}")
            except Exception as e:
                print(f"Error deleting {cp['filename']}: {e}")
    
    def get_checkpoint_info(self, checkpoint_path: str) -> Dict:
        """Get detailed information about a specific checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            info = {
                'epoch': checkpoint.get('epoch', 0),
                'train_losses': checkpoint.get('train_losses', []),
                'val_losses': checkpoint.get('val_losses', []),
                'contrastive_losses': checkpoint.get('contrastive_losses', []),
                'reconstruction_losses': checkpoint.get('reconstruction_losses', []),
                'contrastive_weight': checkpoint.get('contrastive_weight', 1.0),
                'reconstruction_weight': checkpoint.get('reconstruction_weight', 1.0),
                'has_scheduler': checkpoint.get('scheduler_state_dict') is not None,
                'has_optimizer': checkpoint.get('optimizer_state_dict') is not None,
                'file_size_mb': os.path.getsize(checkpoint_path) / (1024 * 1024)
            }
            
            return info
        except Exception as e:
            return {'error': str(e)}


def main():
    """Main function for checkpoint manager CLI"""
    parser = argparse.ArgumentParser(description='Checkpoint Manager for Contrastive Learning Model')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory containing checkpoints')
    parser.add_argument('--list', action='store_true',
                       help='List all available checkpoints')
    parser.add_argument('--latest', action='store_true',
                       help='Show latest checkpoint path')
    parser.add_argument('--epoch', type=int, default=None,
                       help='Get checkpoint for specific epoch')
    parser.add_argument('--info', type=str, default=None,
                       help='Get detailed info about specific checkpoint')
    parser.add_argument('--clean', type=int, default=None,
                       help='Clean old checkpoints, keeping only last N')
    
    args = parser.parse_args()
    
    manager = CheckpointManager(args.save_dir)
    
    if args.list:
        manager.print_checkpoints()
    elif args.latest:
        latest = manager.get_latest_checkpoint()
        if latest:
            print(f"Latest checkpoint: {latest}")
        else:
            print("No checkpoints found")
    elif args.epoch is not None:
        checkpoint = manager.get_checkpoint_by_epoch(args.epoch)
        if checkpoint:
            print(f"Checkpoint for epoch {args.epoch}: {checkpoint}")
        else:
            print(f"No checkpoint found for epoch {args.epoch}")
    elif args.info:
        info = manager.get_checkpoint_info(args.info)
        if 'error' in info:
            print(f"Error loading checkpoint: {info['error']}")
        else:
            print(f"\nCheckpoint Info: {args.info}")
            print("-" * 40)
            print(f"Epoch: {info['epoch'] + 1}")
            print(f"File size: {info['file_size_mb']:.1f} MB")
            print(f"Has scheduler: {info['has_scheduler']}")
            print(f"Has optimizer: {info['has_optimizer']}")
            print(f"Train losses: {len(info['train_losses'])}")
            print(f"Val losses: {len(info['val_losses'])}")
            if info['train_losses']:
                print(f"Latest train loss: {info['train_losses'][-1]:.4f}")
    elif args.clean is not None:
        manager.clean_old_checkpoints(args.clean)
    else:
        manager.print_checkpoints()


if __name__ == "__main__":
    main()
