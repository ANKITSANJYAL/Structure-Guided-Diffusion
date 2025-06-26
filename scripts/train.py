#!/usr/bin/env python3
"""
Training script for GeoDreamer - Geometry-Aware Diffusion Model.

Usage:
    python scripts/train.py --config configs/training_config.yaml
    python scripts/train.py --config configs/training_config.yaml --resume models/checkpoints/checkpoint_epoch_50.pt
"""

import argparse
import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import wandb
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import our modules
from models.diffusion_model import create_geometry_aware_diffusion
from data.dataset import create_dataset
from training.trainer import DiffusionTrainer
from training.losses import DiffusionLoss
from training.metrics import compute_metrics
from utils.logging_utils import setup_logging, setup_wandb


def download_pretrained_models():
    """Download required pre-trained models (DINOv2 and CLIP)."""
    print("Downloading pre-trained models...")
    
    try:
        # Import here to avoid issues if not installed
        from models.geometry_encoder import GeometryEncoder
        from data.semantic_conditioning import SemanticConditioner
        
        # Download DINOv2
        print("Downloading DINOv2 model...")
        geometry_encoder = GeometryEncoder(model_name="dinov2_vitb14", freeze_backbone=True)
        test_image = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = geometry_encoder(test_image)
        print("✅ DINOv2 model downloaded successfully!")
        
        # Download CLIP
        print("Downloading CLIP model...")
        semantic_conditioner = SemanticConditioner(model_name="openai/clip-vit-base-patch32", freeze_backbone=True)
        test_text = ["daisy", "rose"]
        with torch.no_grad():
            embeddings = semantic_conditioner(test_text)
        print("✅ CLIP model downloaded successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to download pre-trained models: {e}")
        print("Please run: python scripts/download_models.py")
        return False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train GeoDreamer model')
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--wandb_project',
        type=str,
        default='geodreamer',
        help='WandB project name'
    )
    
    parser.add_argument(
        '--wandb_run_name',
        type=str,
        default=None,
        help='WandB run name'
    )
    
    parser.add_argument(
        '--no_wandb',
        action='store_true',
        help='Disable WandB logging'
    )
    
    parser.add_argument(
        '--skip_model_download',
        action='store_true',
        help='Skip downloading pre-trained models'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_device(config: dict) -> torch.device:
    """Setup device for training."""
    if config['hardware']['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config['hardware']['device'])
    
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device


def create_dataloaders(config: dict):
    """Create training and validation dataloaders."""
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(config['data']['image_size']),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config['data']['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create datasets
    train_dataset = create_dataset(
        dataset_name=config['data']['dataset_name'],
        root_dir=config['data']['root_dir'],
        split='train',
        transform=train_transform,
    )
    
    val_dataset = create_dataset(
        dataset_name=config['data']['dataset_name'],
        root_dir=config['data']['root_dir'],
        split='val',
        transform=val_transform,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=False,
    )
    
    return train_loader, val_loader


def create_model(config: dict, device: torch.device):
    """Create the geometry-aware diffusion model."""
    
    # Create model
    model = create_geometry_aware_diffusion(config)
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


def create_optimizer_and_scheduler(model, config: dict):
    """Create optimizer and learning rate scheduler."""
    
    # Optimizer
    if config['training']['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['training']['optimizer']}")
    
    # Scheduler
    if config['training']['scheduler'] == 'cosine':
        total_steps = len(train_loader) * config['training']['epochs']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=1e-6
        )
    else:
        scheduler = None
    
    return optimizer, scheduler


def load_checkpoint(model, optimizer, scheduler, checkpoint_path: str, device: torch.device):
    """Load model checkpoint."""
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if 'scheduler_state_dict' in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Get starting epoch
    start_epoch = checkpoint.get('epoch', 0) + 1
    
    print(f"Resumed from epoch {start_epoch}")
    
    return start_epoch


def main():
    """Main training function."""
    
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    device = setup_device(config)
    
    # Download pre-trained models if not skipped
    if not args.skip_model_download:
        if not download_pretrained_models():
            print("Failed to download pre-trained models. Exiting.")
            return
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    
    # Setup WandB
    if not args.no_wandb:
        wandb_run = setup_wandb(
            project=args.wandb_project,
            run_name=args.wandb_run_name,
            config=config
        )
    else:
        wandb_run = None
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'samples'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    
    # Create model
    print("Creating model...")
    model = create_model(config, device)
    
    # Create optimizer and scheduler
    print("Creating optimizer and scheduler...")
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, scheduler, args.resume, device)
    
    # Create trainer
    print("Creating trainer...")
    trainer = DiffusionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        output_dir=args.output_dir,
        wandb_run=wandb_run,
    )
    
    # Start training
    print("Starting training...")
    trainer.train(start_epoch=start_epoch)
    
    print("Training completed!")
    
    # Save final model
    final_checkpoint_path = os.path.join(args.output_dir, 'checkpoints', 'final_model.pt')
    trainer.save_checkpoint(final_checkpoint_path, is_final=True)
    
    print(f"Final model saved to {final_checkpoint_path}")


if __name__ == '__main__':
    main() 