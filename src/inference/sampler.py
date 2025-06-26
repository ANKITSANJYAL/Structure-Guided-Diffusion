"""
Inference sampler for GeoDreamer diffusion model.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


class GeometryAwareSampler:
    """
    Sampler for geometry-aware diffusion model inference.
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any], device: torch.device):
        """
        Initialize the sampler.
        
        Args:
            model: The trained diffusion model
            config: Configuration dictionary
            device: Device to use for inference
        """
        self.model = model
        self.config = config
        self.device = device
        
        # Set model to evaluation mode
        self.model.eval()
    
    def sample(
        self,
        class_labels: Optional[torch.Tensor] = None,
        geometry_images: Optional[torch.Tensor] = None,
        num_samples: int = 1,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        geometry_guidance_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate samples using the diffusion model.
        
        Args:
            class_labels: Class labels for semantic conditioning
            geometry_images: Images for geometry conditioning
            num_samples: Number of samples to generate
            num_inference_steps: Number of denoising steps
            guidance_scale: Scale for classifier-free guidance
            geometry_guidance_scale: Scale for geometry guidance
            
        Returns:
            Generated images
        """
        # Use default values if not provided
        if num_inference_steps is None:
            num_inference_steps = self.config['sampling']['num_inference_steps']
        if guidance_scale is None:
            guidance_scale = self.config['sampling']['guidance_scale']
        if geometry_guidance_scale is None:
            geometry_guidance_scale = self.config['sampling']['geometry_guidance_scale']
        
        with torch.no_grad():
            samples = self.model.sample(
                batch_size=num_samples,
                class_labels=class_labels,
                geometry_images=geometry_images,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                geometry_guidance_scale=geometry_guidance_scale,
                device=self.device,
            )
        
        return samples
    
    def denormalize_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Denormalize images from model output to [0, 1] range.
        
        Args:
            images: Normalized images from model
            
        Returns:
            Denormalized images
        """
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
        denormalized = images * std + mean
        denormalized = torch.clamp(denormalized, 0, 1)
        
        return denormalized
    
    def generate_flower_samples(
        self,
        class_indices: List[int],
        num_samples_per_class: int = 1,
        use_geometry: bool = False,
        geometry_image_path: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate samples for specific flower classes.
        
        Args:
            class_indices: List of class indices to generate
            num_samples_per_class: Number of samples per class
            use_geometry: Whether to use geometry conditioning
            geometry_image_path: Path to geometry reference image
            
        Returns:
            Dictionary containing generated samples and metadata
        """
        all_samples = []
        all_labels = []
        
        # Load geometry image if provided
        geometry_image = None
        if use_geometry and geometry_image_path:
            geometry_image = self._load_geometry_image(geometry_image_path)
        
        for class_idx in class_indices:
            # Create labels for this class
            labels = torch.full((num_samples_per_class,), class_idx, device=self.device)
            
            # Prepare geometry conditioning
            geometry_batch = None
            if use_geometry and geometry_image is not None:
                geometry_batch = geometry_image.repeat(num_samples_per_class, 1, 1, 1)
            
            # Generate samples
            samples = self.sample(
                class_labels=labels,
                geometry_images=geometry_batch,
                num_samples=num_samples_per_class,
            )
            
            all_samples.append(samples)
            all_labels.extend([f"class_{class_idx}"] * num_samples_per_class)
        
        # Concatenate all samples
        if all_samples:
            all_samples = torch.cat(all_samples, dim=0)
            all_samples = self.denormalize_images(all_samples)
        
        return {
            'samples': all_samples,
            'labels': all_labels,
            'class_indices': class_indices,
        }
    
    def _load_geometry_image(self, image_path: str) -> torch.Tensor:
        """
        Load and preprocess a geometry reference image.
        
        Args:
            image_path: Path to the reference image
            
        Returns:
            Preprocessed image tensor
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Resize to model input size
        image_size = self.config['data']['image_size']
        image = image.resize((image_size, image_size))
        
        # Convert to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        image_tensor = image_tensor.to(self.device)
        
        return image_tensor
    
    def generate_comparison(
        self,
        class_idx: int,
        num_samples: int = 4,
        geometry_image_path: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate comparison between with and without geometry conditioning.
        
        Args:
            class_idx: Class index to generate
            num_samples: Number of samples to generate
            geometry_image_path: Path to geometry reference image
            
        Returns:
            Dictionary containing samples with and without geometry
        """
        # Generate without geometry
        labels = torch.full((num_samples,), class_idx, device=self.device)
        
        samples_no_geometry = self.sample(
            class_labels=labels,
            geometry_images=None,
            num_samples=num_samples,
        )
        samples_no_geometry = self.denormalize_images(samples_no_geometry)
        
        # Generate with geometry
        samples_with_geometry = None
        if geometry_image_path:
            geometry_image = self._load_geometry_image(geometry_image_path)
            geometry_batch = geometry_image.repeat(num_samples, 1, 1, 1)
            
            samples_with_geometry = self.sample(
                class_labels=labels,
                geometry_images=geometry_batch,
                num_samples=num_samples,
            )
            samples_with_geometry = self.denormalize_images(samples_with_geometry)
        
        return {
            'without_geometry': samples_no_geometry,
            'with_geometry': samples_with_geometry,
            'class_idx': class_idx,
        }
    
    def generate_grid(
        self,
        class_indices: List[int],
        num_samples_per_class: int = 4,
        use_geometry: bool = False,
        geometry_image_path: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Generate a grid of samples for visualization.
        
        Args:
            class_indices: List of class indices
            num_samples_per_class: Number of samples per class
            use_geometry: Whether to use geometry conditioning
            geometry_image_path: Path to geometry reference image
            
        Returns:
            Grid of generated images
        """
        # Generate samples
        result = self.generate_flower_samples(
            class_indices=class_indices,
            num_samples_per_class=num_samples_per_class,
            use_geometry=use_geometry,
            geometry_image_path=geometry_image_path,
        )
        
        samples = result['samples']
        
        # Create grid
        import torchvision.utils as vutils
        grid = vutils.make_grid(
            samples, 
            nrow=num_samples_per_class, 
            padding=2, 
            normalize=False
        )
        
        return grid
    
    def save_samples(
        self,
        samples: torch.Tensor,
        output_path: str,
        labels: Optional[List[str]] = None,
    ):
        """
        Save generated samples to disk.
        
        Args:
            samples: Generated samples
            output_path: Output file path
            labels: Optional labels for the samples
        """
        import torchvision.utils as vutils
        
        # Create grid if multiple samples
        if samples.dim() == 4 and samples.size(0) > 1:
            grid = vutils.make_grid(samples, nrow=4, padding=2, normalize=False)
            vutils.save_image(grid, output_path)
        else:
            # Single image
            vutils.save_image(samples, output_path)
        
        print(f"Samples saved to {output_path}")


def create_sampler(model: nn.Module, config: Dict[str, Any], device: torch.device) -> GeometryAwareSampler:
    """
    Create a geometry-aware sampler.
    
    Args:
        model: The trained diffusion model
        config: Configuration dictionary
        device: Device to use for inference
        
    Returns:
        GeometryAwareSampler instance
    """
    return GeometryAwareSampler(model, config, device) 