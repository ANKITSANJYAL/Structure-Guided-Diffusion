#!/usr/bin/env python3
"""
Generation script for GeoDreamer - Geometry-Aware Diffusion Model.

Usage:
    python scripts/generate.py --checkpoint models/checkpoints/best_model.pt --class_idx 5 --num_samples 4
    python scripts/generate.py --checkpoint models/checkpoints/best_model.pt --geometry_image path/to/image.jpg --class_idx 10
"""

import argparse
import os
import sys
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import our modules
from models.diffusion_model import create_geometry_aware_diffusion
from inference.sampler import create_sampler
from utils.logging_utils import load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate samples with GeoDreamer')
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--class_idx',
        type=int,
        default=0,
        help='Class index to generate'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=4,
        help='Number of samples to generate'
    )
    
    parser.add_argument(
        '--geometry_image',
        type=str,
        default=None,
        help='Path to geometry reference image'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results/generated',
        help='Output directory for generated images'
    )
    
    parser.add_argument(
        '--guidance_scale',
        type=float,
        default=7.5,
        help='Classifier-free guidance scale'
    )
    
    parser.add_argument(
        '--num_inference_steps',
        type=int,
        default=50,
        help='Number of inference steps'
    )
    
    parser.add_argument(
        '--save_grid',
        action='store_true',
        help='Save images as a grid'
    )
    
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device):
    """Load the trained model from checkpoint."""
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create model
    model = create_geometry_aware_diffusion(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)
    
    print(f"Model loaded successfully!")
    print(f"Training epoch: {checkpoint['epoch']}")
    print(f"Training loss: {checkpoint['val_loss']:.4f}")
    
    return model, config


def setup_device() -> torch.device:
    """Setup device for generation."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    return device


def load_geometry_image(image_path: str, image_size: int = 224, device: torch.device = None):
    """Load and preprocess geometry reference image."""
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Resize
    image = image.resize((image_size, image_size))
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)
    
    return image_tensor, image


def generate_samples(
    model,
    config: dict,
    class_idx: int,
    num_samples: int,
    geometry_image: torch.Tensor = None,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    device: torch.device = None,
):
    """Generate samples using the model."""
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create labels
    labels = torch.full((num_samples,), class_idx, device=device)
    
    # Prepare geometry conditioning
    geometry_batch = None
    if geometry_image is not None:
        geometry_batch = geometry_image.repeat(num_samples, 1, 1, 1)
    
    print(f"Generating {num_samples} samples for class {class_idx}")
    if geometry_batch is not None:
        print("Using geometry conditioning")
    else:
        print("No geometry conditioning")
    
    with torch.no_grad():
        # Generate samples
        samples = model.sample(
            batch_size=num_samples,
            class_labels=labels,
            geometry_images=geometry_batch,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            device=device,
        )
    
    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    samples = samples * std + mean
    samples = torch.clamp(samples, 0, 1)
    
    return samples


def save_samples(
    samples: torch.Tensor,
    output_dir: str,
    class_idx: int,
    save_grid: bool = True,
):
    """Save generated samples."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    if save_grid and samples.size(0) > 1:
        # Save as grid
        import torchvision.utils as vutils
        grid = vutils.make_grid(samples, nrow=4, padding=2, normalize=False)
        grid_path = os.path.join(output_dir, f'class_{class_idx}_grid.png')
        vutils.save_image(grid, grid_path)
        print(f"Grid saved to {grid_path}")
        
        # Also save individual images
        for i in range(samples.size(0)):
            sample_path = os.path.join(output_dir, f'class_{class_idx}_sample_{i+1}.png')
            vutils.save_image(samples[i], sample_path)
    else:
        # Save individual images
        import torchvision.utils as vutils
        for i in range(samples.size(0)):
            sample_path = os.path.join(output_dir, f'class_{class_idx}_sample_{i+1}.png')
            vutils.save_image(samples[i], sample_path)
            print(f"Sample {i+1} saved to {sample_path}")


def visualize_samples(samples: torch.Tensor, class_idx: int, geometry_image: Image = None):
    """Visualize generated samples."""
    
    num_samples = samples.size(0)
    
    if geometry_image:
        # Show geometry image and samples
        fig, axes = plt.subplots(1, num_samples + 1, figsize=(4 * (num_samples + 1), 4))
        
        # Show geometry image
        axes[0].imshow(geometry_image)
        axes[0].set_title("Geometry Reference")
        axes[0].axis('off')
        
        # Show generated samples
        for i in range(num_samples):
            sample = samples[i].cpu().permute(1, 2, 0).numpy()
            axes[i + 1].imshow(sample)
            axes[i + 1].set_title(f"Generated {i+1}")
            axes[i + 1].axis('off')
    else:
        # Show only generated samples
        fig, axes = plt.subplots(1, num_samples, figsize=(4 * num_samples, 4))
        
        if num_samples == 1:
            axes = [axes]
        
        for i in range(num_samples):
            sample = samples[i].cpu().permute(1, 2, 0).numpy()
            axes[i].imshow(sample)
            axes[i].set_title(f"Class {class_idx} - Sample {i+1}")
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    """Main generation function."""
    
    # Parse arguments
    args = parse_args()
    
    # Setup device
    device = setup_device()
    
    # Load model
    model, config = load_model(args.checkpoint, device)
    
    # Load geometry image if provided
    geometry_image_tensor = None
    geometry_image_pil = None
    if args.geometry_image:
        if os.path.exists(args.geometry_image):
            geometry_image_tensor, geometry_image_pil = load_geometry_image(
                args.geometry_image, 
                config['data']['image_size'],
                device
            )
            print(f"Loaded geometry image: {args.geometry_image}")
        else:
            print(f"Warning: Geometry image not found: {args.geometry_image}")
    
    # Generate samples
    samples = generate_samples(
        model=model,
        config=config,
        class_idx=args.class_idx,
        num_samples=args.num_samples,
        geometry_image=geometry_image_tensor,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        device=device,
    )
    
    # Save samples
    save_samples(
        samples=samples,
        output_dir=args.output_dir,
        class_idx=args.class_idx,
        save_grid=args.save_grid,
    )
    
    # Visualize samples
    visualize_samples(samples, args.class_idx, geometry_image_pil)
    
    print(f"\nGeneration completed!")
    print(f"Samples saved to: {args.output_dir}")


if __name__ == '__main__':
    main() 