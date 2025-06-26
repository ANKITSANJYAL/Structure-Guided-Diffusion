"""
Geometry-aware diffusion model with DINO-based geometry conditioning and CLIP semantic conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import math

from .unet import UNetModel
from .geometry_encoder import GeometryEncoder, GeometryTokenProcessor
from ..data.semantic_conditioning import SemanticConditioner


class GeometryAwareDiffusion(nn.Module):
    """
    Geometry-aware diffusion model that conditions on both semantic (CLIP) and geometric (DINO) features.
    
    This model extends the standard DDPM with:
    1. Semantic conditioning using CLIP embeddings
    2. Geometry conditioning using DINO patch tokens
    3. Cross-attention mechanisms to incorporate both types of conditioning
    """
    
    def __init__(
        self,
        unet_config: Dict[str, Any],
        geometry_encoder_config: Dict[str, Any],
        semantic_conditioner_config: Dict[str, Any],
        diffusion_config: Dict[str, Any],
    ):
        """
        Initialize the geometry-aware diffusion model.
        
        Args:
            unet_config: Configuration for the U-Net model
            geometry_encoder_config: Configuration for the geometry encoder
            semantic_conditioner_config: Configuration for the semantic conditioner
            diffusion_config: Configuration for the diffusion process
        """
        super().__init__()
        
        # Diffusion parameters
        self.diffusion_steps = diffusion_config.get('diffusion_steps', 1000)
        self.beta_start = diffusion_config.get('beta_start', 0.0001)
        self.beta_end = diffusion_config.get('beta_end', 0.02)
        self.schedule = diffusion_config.get('schedule', 'linear')
        
        # Initialize noise schedule
        self.betas = self._get_noise_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Model components
        self.unet = UNetModel(**unet_config)
        self.geometry_encoder = GeometryEncoder(**geometry_encoder_config)
        self.semantic_conditioner = SemanticConditioner(**semantic_conditioner_config)
        self.geometry_processor = GeometryTokenProcessor(self.geometry_encoder)
        
        # Optional projection layers for feature alignment
        self.semantic_projection = nn.Linear(
            self.semantic_conditioner.get_feature_dim(),
            unet_config.get('context_dim', 768)
        )
        self.geometry_projection = nn.Linear(
            self.geometry_encoder.get_feature_dim(),
            unet_config.get('geometry_dim', 768)
        )
    
    def _get_noise_schedule(self) -> torch.Tensor:
        """Get the noise schedule for the diffusion process."""
        if self.schedule == 'linear':
            return torch.linspace(self.beta_start, self.beta_end, self.diffusion_steps)
        elif self.schedule == 'cosine':
            return self._cosine_beta_schedule()
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")
    
    def _cosine_beta_schedule(self) -> torch.Tensor:
        """Cosine beta schedule as proposed in Improved DDPM."""
        steps = self.diffusion_steps + 1
        x = torch.linspace(0, self.diffusion_steps, steps)
        alphas_cumprod = torch.cos(((x / self.diffusion_steps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        geometry_images: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the diffusion model.
        
        Args:
            x: Noisy images of shape (B, C, H, W)
            t: Timesteps of shape (B,)
            class_labels: Class labels for semantic conditioning
            geometry_images: Images for geometry conditioning
            return_dict: Whether to return a dictionary
            
        Returns:
            Dictionary containing model predictions
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Ensure t is on the correct device
        t = t.to(device)
        
        # Get semantic conditioning
        if class_labels is not None:
            semantic_embeddings = self._get_semantic_conditioning(class_labels)
            semantic_embeddings = self.semantic_projection(semantic_embeddings)
        else:
            semantic_embeddings = None
        
        # Get geometry conditioning
        if geometry_images is not None:
            geometry_tokens = self._get_geometry_conditioning(geometry_images)
            geometry_tokens = self.geometry_projection(geometry_tokens)
        else:
            geometry_tokens = None
        
        # Predict noise using U-Net
        noise_pred = self.unet(
            x,
            t,
            context=semantic_embeddings,
            geometry_tokens=geometry_tokens,
        )
        
        if return_dict:
            return {
                'noise_pred': noise_pred,
                'semantic_embeddings': semantic_embeddings,
                'geometry_tokens': geometry_tokens,
            }
        else:
            return noise_pred
    
    def _get_semantic_conditioning(self, class_labels: torch.Tensor) -> torch.Tensor:
        """Get semantic conditioning from class labels."""
        # Convert class indices to class names (assuming Oxford Flowers 102)
        class_names = self._get_class_names_from_indices(class_labels)
        
        # Get CLIP embeddings
        semantic_embeddings = self.semantic_conditioner(class_names)
        return semantic_embeddings
    
    def _get_geometry_conditioning(self, geometry_images: torch.Tensor) -> torch.Tensor:
        """Get geometry conditioning from images."""
        # Extract geometry tokens
        geometry_tokens = self.geometry_processor.process_for_diffusion(geometry_images)
        return geometry_tokens
    
    def _get_class_names_from_indices(self, class_indices: torch.Tensor) -> list:
        """Convert class indices to class names."""
        # Oxford Flowers 102 class names (partial list for brevity)
        class_names = [
            "pink primrose", "hard-leaved pocket orchid", "canterbury bells",
            "sweet pea", "english marigold", "tiger lily", "moon orchid",
            "bird of paradise", "monkshood", "globe thistle", "snapdragon",
            # ... (full list would be included)
        ]
        
        # Convert indices to names
        names = []
        for idx in class_indices:
            if idx < len(class_names):
                names.append(class_names[idx])
            else:
                names.append("unknown")
        
        return names
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample from q(x_t | x_0) for training.
        
        Args:
            x_start: Original images
            t: Timesteps
            noise: Optional noise to use
            
        Returns:
            Noisy images at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Get alpha_cumprod for timesteps t
        alphas_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        
        # Sample from q(x_t | x_0)
        x_t = torch.sqrt(alphas_cumprod_t) * x_start + torch.sqrt(1 - alphas_cumprod_t) * noise
        
        return x_t
    
    def p_losses(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        geometry_images: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses.
        
        Args:
            x_start: Original images
            t: Timesteps
            class_labels: Class labels for semantic conditioning
            geometry_images: Images for geometry conditioning
            noise: Optional noise to use
            
        Returns:
            Dictionary containing losses
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Add noise to images
        x_noisy = self.q_sample(x_start, t, noise)
        
        # Predict noise
        noise_pred = self.forward(
            x_noisy,
            t,
            class_labels=class_labels,
            geometry_images=geometry_images,
            return_dict=False,
        )
        
        # Compute loss
        loss = F.mse_loss(noise_pred, noise, reduction='mean')
        
        return {
            'loss': loss,
            'noise_pred': noise_pred,
            'noise_target': noise,
        }
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        class_labels: Optional[torch.Tensor] = None,
        geometry_images: Optional[torch.Tensor] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        geometry_guidance_scale: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Generate samples using the diffusion model.
        
        Args:
            batch_size: Number of samples to generate
            class_labels: Class labels for semantic conditioning
            geometry_images: Images for geometry conditioning
            num_inference_steps: Number of denoising steps
            guidance_scale: Scale for classifier-free guidance
            geometry_guidance_scale: Scale for geometry guidance
            device: Device to use for generation
            
        Returns:
            Generated images
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Initialize with noise
        x = torch.randn(batch_size, 3, 224, 224, device=device)
        
        # Setup timesteps
        timesteps = torch.linspace(0, self.diffusion_steps - 1, num_inference_steps, dtype=torch.long, device=device)
        timesteps = timesteps.flip(0)  # Reverse order for denoising
        
        # Denoising loop
        for i, t in enumerate(timesteps):
            t_batch = t.repeat(batch_size)
            
            # Predict noise
            noise_pred = self.forward(
                x,
                t_batch,
                class_labels=class_labels,
                geometry_images=geometry_images,
                return_dict=False,
            )
            
            # Apply guidance if specified
            if guidance_scale > 1.0:
                # Unconditional prediction
                noise_pred_uncond = self.forward(
                    x,
                    t_batch,
                    class_labels=None,
                    geometry_images=None,
                    return_dict=False,
                )
                
                # Classifier-free guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
            
            # Denoising step
            x = self._denoising_step(x, t, noise_pred)
        
        return x
    
    def _denoising_step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        noise_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Perform a single denoising step."""
        # Get alpha values
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        alpha_cumprod_prev_t = self.alphas_cumprod_prev[t].view(-1, 1, 1, 1)
        
        # Compute predicted x_0
        pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
        
        # Compute variance
        variance = 0
        if t > 0:
            noise = torch.randn_like(x)
            variance = torch.sqrt((1 - alpha_cumprod_prev_t) / (1 - alpha_cumprod_t)) * noise
        
        # Compute x_{t-1}
        x_prev = torch.sqrt(alpha_cumprod_prev_t) * pred_x0 + variance
        
        return x_prev


def create_geometry_aware_diffusion(config: Dict[str, Any]) -> GeometryAwareDiffusion:
    """
    Create a geometry-aware diffusion model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        GeometryAwareDiffusion instance
    """
    return GeometryAwareDiffusion(
        unet_config=config['model']['unet'],
        geometry_encoder_config=config['model']['geometry_encoder'],
        semantic_conditioner_config=config['model']['semantic_conditioning'],
        diffusion_config={
            'diffusion_steps': config['model']['diffusion_steps'],
            'beta_start': config['model']['beta_start'],
            'beta_end': config['model']['beta_end'],
            'schedule': config['model']['schedule'],
        },
    ) 