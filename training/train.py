"""
Main training script for Structure-Guided-Diffusion
Supports config-driven training for different phases
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
from tqdm import tqdm
import logging
from pathlib import Path

# Add src to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from conditioning.clip_encoder import create_clip_encoder
from models.unet import create_unet


class DiffusionTrainer:
    """Trainer for diffusion models with CLIP conditioning."""
    
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.setup_logging()
        
        # Initialize models
        self.clip_encoder = create_clip_encoder(self.config).to(self.device)
        self.unet = create_unet(self.config).to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.unet.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Setup diffusion scheduler
        self.setup_diffusion()
        
        # Setup data
        self.setup_data()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
    def load_config(self, config_path):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path(self.config['paths']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_diffusion(self):
        """Setup diffusion scheduler."""
        diffusion_config = self.config['diffusion']
        
        # Simple linear beta schedule
        self.num_timesteps = diffusion_config['num_train_timesteps']
        self.beta_start = diffusion_config['beta_start']
        self.beta_end = diffusion_config['beta_end']
        
        # Create beta schedule
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # Precompute values
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
    
    def setup_data(self):
        """Setup data loaders."""
        # For now, we'll use a placeholder dataset
        # TODO: Implement Oxford Flowers 102 dataset
        self.logger.info("Setting up data loaders...")
        
        # Placeholder for now
        self.train_loader = None
        self.val_loader = None
    
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise
    
    def train_step(self, batch):
        """Single training step."""
        images, texts = batch
        images = images.to(self.device)
        
        batch_size = images.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
        
        # Add noise to images
        noisy_images, noise = self.q_sample(images, t)
        
        # Encode text with CLIP
        text_embeddings = self.clip_encoder(texts)
        
        # Classifier-free guidance: randomly drop text conditioning
        cfg_prob = self.config['training']['cfg_probability']
        if torch.rand(1) < cfg_prob:
            text_embeddings = torch.zeros_like(text_embeddings)
        
        # Predict noise
        predicted_noise = self.unet(noisy_images, t, text_embeddings)
        
        # Compute loss
        loss = nn.MSELoss()(predicted_noise, noise)
        
        return loss
    
    def train_epoch(self):
        """Train for one epoch."""
        self.unet.train()
        self.clip_encoder.eval()  # CLIP is frozen
        
        total_loss = 0
        num_batches = 0
        
        # Placeholder training loop
        # TODO: Replace with actual data loader
        for i in range(10):  # Placeholder: 10 batches
            # Placeholder batch
            batch_size = self.config['training']['batch_size']
            images = torch.randn(batch_size, 3, 256, 256).to(self.device)
            texts = ["a beautiful flower"] * batch_size
            
            batch = (images, texts)
            
            # Training step
            self.optimizer.zero_grad()
            loss = self.train_step(batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.unet.parameters(), 
                self.config['training']['gradient_clip_val']
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Logging
            if self.global_step % self.config['training']['log_every_n_steps'] == 0:
                self.logger.info(f"Step {self.global_step}: Loss = {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        self.logger.info(f"Epoch {self.epoch}: Average Loss = {avg_loss:.4f}")
        
        return avg_loss
    
    def save_checkpoint(self):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config['paths']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'unet_state_dict': self.unet.state_dict(),
            'clip_encoder_state_dict': self.clip_encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }
        
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{self.epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Config: {self.config['experiment']['name']}")
        
        num_epochs = self.config['training']['num_epochs']
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train epoch
            loss = self.train_epoch()
            
            # Save checkpoint
            if (epoch + 1) % self.config['training']['save_every_n_epochs'] == 0:
                self.save_checkpoint()
            
            # Evaluation
            if (epoch + 1) % self.config['training']['eval_every_n_epochs'] == 0:
                # TODO: Implement evaluation
                pass
        
        self.logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description='Train Structure-Guided-Diffusion model')
    parser.add_argument('--config', type=str, default='configs/baseline.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    trainer = DiffusionTrainer(args.config)
    trainer.train()


if __name__ == '__main__':
    main() 