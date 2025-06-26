"""
Training trainer for GeoDreamer diffusion model.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from typing import Optional, Dict, Any
import wandb


class DiffusionTrainer:
    """
    Trainer class for the geometry-aware diffusion model.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        config: Dict[str, Any],
        device: torch.device,
        output_dir: str = "./results",
        wandb_run: Optional[wandb.run] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The diffusion model
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            config: Configuration dictionary
            device: Device to train on
            output_dir: Output directory for results
            wandb_run: WandB run for logging
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.output_dir = output_dir
        self.wandb_run = wandb_run
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        # Setup mixed precision
        self.use_amp = config['hardware']['mixed_precision']
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Create output directories
        os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'samples'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
    
    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Sample random timesteps
            batch_size = images.shape[0]
            t = torch.randint(
                0, 
                self.config['model']['diffusion_steps'], 
                (batch_size,), 
                device=self.device
            )
            
            # Forward pass with mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    loss_dict = self.model.p_losses(
                        x_start=images,
                        t=t,
                        class_labels=labels,
                        geometry_images=images,  # Use same images for geometry
                    )
                    loss = loss_dict['loss']
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config['training']['gradient_clip_val'] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['training']['gradient_clip_val']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_dict = self.model.p_losses(
                    x_start=images,
                    t=t,
                    class_labels=labels,
                    geometry_images=images,
                )
                loss = loss_dict['loss']
                
                self.optimizer.zero_grad()
                loss.backward()
                
                if self.config['training']['gradient_clip_val'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['training']['gradient_clip_val']
                    )
                
                self.optimizer.step()
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Log to wandb
            if (self.wandb_run and 
                batch_idx % self.config['training']['log_every_n_steps'] == 0):
                self.wandb_run.log({
                    'train/loss': loss.item(),
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'train/step': epoch * num_batches + batch_idx,
                })
        
        return total_loss / num_batches
    
    def validate(self, epoch: int) -> float:
        """
        Validate the model.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Sample random timesteps
                batch_size = images.shape[0]
                t = torch.randint(
                    0, 
                    self.config['model']['diffusion_steps'], 
                    (batch_size,), 
                    device=self.device
                )
                
                # Forward pass
                loss_dict = self.model.p_losses(
                    x_start=images,
                    t=t,
                    class_labels=labels,
                    geometry_images=images,
                )
                loss = loss_dict['loss']
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        
        # Log to wandb
        if self.wandb_run:
            self.wandb_run.log({
                'val/loss': avg_loss,
                'val/epoch': epoch,
            })
        
        return avg_loss
    
    def generate_samples(self, epoch: int, num_samples: int = 8):
        """
        Generate sample images for visualization.
        
        Args:
            epoch: Current epoch number
            num_samples: Number of samples to generate
        """
        self.model.eval()
        
        # Sample random class labels
        class_labels = torch.randint(0, 102, (num_samples,), device=self.device)
        
        with torch.no_grad():
            # Generate samples
            samples = self.model.sample(
                batch_size=num_samples,
                class_labels=class_labels,
                geometry_images=None,  # No geometry conditioning for now
                num_inference_steps=self.config['sampling']['num_inference_steps'],
                guidance_scale=self.config['sampling']['guidance_scale'],
                device=self.device,
            )
        
        # Denormalize images
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        samples = samples * std + mean
        samples = torch.clamp(samples, 0, 1)
        
        # Save samples
        sample_path = os.path.join(self.output_dir, 'samples', f'epoch_{epoch+1}.png')
        
        # Create grid and save
        grid = vutils.make_grid(samples, nrow=4, padding=2, normalize=False)
        vutils.save_image(grid, sample_path)
        
        # Log to wandb
        if self.wandb_run:
            self.wandb_run.log({
                'samples': wandb.Image(sample_path),
                'epoch': epoch,
            })
        
        print(f"Samples saved to {sample_path}")
        
        return samples
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False, is_final: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            val_loss: Validation loss
            is_best: Whether this is the best model so far
            is_final: Whether this is the final checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'config': self.config,
        }
        
        # Save regular checkpoint
        if not is_final:
            checkpoint_path = os.path.join(
                self.output_dir, 'checkpoints', f'checkpoint_epoch_{epoch+1}.pt'
            )
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.output_dir, 'checkpoints', 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"Best model saved to {best_path}")
        
        # Save final model
        if is_final:
            final_path = os.path.join(self.output_dir, 'checkpoints', 'final_model.pt')
            torch.save(checkpoint, final_path)
            print(f"Final model saved to {final_path}")
    
    def train(self, start_epoch: int = 0):
        """
        Main training loop.
        
        Args:
            start_epoch: Starting epoch number
        """
        print(f"Starting training from epoch {start_epoch}")
        
        for epoch in range(start_epoch, self.config['training']['epochs']):
            self.current_epoch = epoch
            
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{self.config['training']['epochs']}")
            print(f"{'='*50}")
            
            # Training
            train_loss = self.train_epoch(epoch)
            
            # Validation
            val_loss = self.validate(epoch)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            
            # Generate samples
            if (epoch + 1) % self.config['training']['log_images_every_n_epochs'] == 0:
                self.generate_samples(epoch)
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            if ((epoch + 1) % self.config['training']['save_every_n_epochs'] == 0 or 
                is_best):
                self.save_checkpoint(epoch, val_loss, is_best)
            
            # Log to wandb
            if self.wandb_run:
                self.wandb_run.log({
                    'train/epoch_loss': train_loss,
                    'val/epoch_loss': val_loss,
                    'epoch': epoch,
                })
        
        print("\nTraining completed!")
        
        # Save final model
        self.save_checkpoint(
            self.config['training']['epochs']-1, 
            val_loss, 
            is_final=True
        ) 