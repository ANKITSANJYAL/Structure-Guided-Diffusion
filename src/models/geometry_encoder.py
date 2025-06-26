"""
Geometry Encoder using DINOv2 for extracting geometric features from images.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
import dinov2
from transformers import AutoImageProcessor


class GeometryEncoder(nn.Module):
    """
    Geometry encoder using DINOv2 to extract geometric features from images.
    
    This module extracts patch-level geometric tokens from images using a pre-trained
    DINOv2 model, which captures fine-grained structural and geometric information.
    """
    
    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        patch_size: int = 14,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        freeze_backbone: bool = True,
        use_cls_token: bool = False,
        projection_dim: Optional[int] = None,
    ):
        """
        Initialize the geometry encoder.
        
        Args:
            model_name: Name of the DINOv2 model to use
            patch_size: Size of image patches
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            freeze_backbone: Whether to freeze the DINOv2 backbone
            use_cls_token: Whether to include CLS token in output
            projection_dim: Optional projection dimension for output features
        """
        super().__init__()
        
        self.model_name = model_name
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token
        self.projection_dim = projection_dim
        
        # Load pre-trained DINOv2 model
        self.backbone = dinov2.models.build_model(model_name)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        
        # Optional projection layer
        if projection_dim is not None:
            self.projection = nn.Linear(embed_dim, projection_dim)
            self.output_dim = projection_dim
        else:
            self.projection = nn.Identity()
            self.output_dim = embed_dim
    
    def forward(
        self, 
        images: torch.Tensor,
        return_patch_features: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Extract geometric features from images.
        
        Args:
            images: Input images of shape (B, C, H, W)
            return_patch_features: Whether to return patch-level features
            
        Returns:
            Dictionary containing:
                - 'features': Global features (B, embed_dim)
                - 'patch_features': Patch-level features (B, num_patches, embed_dim) if return_patch_features=True
                - 'patch_positions': Patch positions for spatial alignment
        """
        batch_size = images.shape[0]
        
        # Get features from DINOv2 backbone
        with torch.no_grad() if not self.training else torch.enable_grad():
            features = self.backbone.forward_features(images)
        
        # Extract patch features (excluding CLS token)
        if hasattr(features, 'x_norm_clstoken'):
            # DINOv2 specific output format
            global_features = features.x_norm_clstoken  # (B, embed_dim)
            patch_features = features.x_norm_patchtokens  # (B, num_patches, embed_dim)
        else:
            # Fallback for other formats
            if isinstance(features, dict):
                global_features = features.get('x_norm_clstoken', features.get('cls_token'))
                patch_features = features.get('x_norm_patchtokens', features.get('patch_tokens'))
            else:
                # Assume features is a tensor
                global_features = features[:, 0]  # CLS token
                patch_features = features[:, 1:]  # Patch tokens
        
        # Apply projection if specified
        global_features = self.projection(global_features)
        patch_features = self.projection(patch_features)
        
        # Calculate patch positions for spatial alignment
        h, w = images.shape[2], images.shape[3]
        num_patches_h = h // self.patch_size
        num_patches_w = w // self.patch_size
        
        # Create patch position grid
        patch_positions = self._create_patch_positions(num_patches_h, num_patches_w)
        patch_positions = patch_positions.unsqueeze(0).expand(batch_size, -1, -1)
        
        output = {
            'features': global_features,
            'patch_positions': patch_positions,
        }
        
        if return_patch_features:
            output['patch_features'] = patch_features
        
        return output
    
    def _create_patch_positions(self, num_patches_h: int, num_patches_w: int) -> torch.Tensor:
        """Create patch position coordinates for spatial alignment."""
        y_coords = torch.arange(num_patches_h, dtype=torch.float32)
        x_coords = torch.arange(num_patches_w, dtype=torch.float32)
        
        # Create grid
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Flatten and stack
        positions = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        
        # Normalize to [0, 1]
        positions[:, 0] /= (num_patches_w - 1)
        positions[:, 1] /= (num_patches_h - 1)
        
        return positions
    
    def get_feature_dim(self) -> int:
        """Get the output feature dimension."""
        return self.output_dim
    
    def get_patch_size(self) -> int:
        """Get the patch size."""
        return self.patch_size


class GeometryTokenProcessor:
    """
    Utility class for processing geometry tokens for diffusion model conditioning.
    """
    
    def __init__(self, geometry_encoder: GeometryEncoder):
        """
        Initialize the processor.
        
        Args:
            geometry_encoder: The geometry encoder instance
        """
        self.geometry_encoder = geometry_encoder
    
    def process_for_diffusion(
        self, 
        images: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Process geometry tokens for diffusion model conditioning.
        
        Args:
            images: Input images
            target_size: Target size for geometry tokens (H, W)
            
        Returns:
            Processed geometry tokens ready for diffusion conditioning
        """
        # Extract geometry features
        geometry_output = self.geometry_encoder(images, return_patch_features=True)
        patch_features = geometry_output['patch_features']  # (B, num_patches, embed_dim)
        
        # Reshape to spatial format if target size is specified
        if target_size is not None:
            target_h, target_w = target_size
            batch_size, num_patches, embed_dim = patch_features.shape
            
            # Calculate current spatial dimensions
            current_h = int((num_patches ** 0.5))
            current_w = current_h
            
            # Reshape to spatial format
            patch_features = patch_features.view(batch_size, current_h, current_w, embed_dim)
            patch_features = patch_features.permute(0, 3, 1, 2)  # (B, embed_dim, H, W)
            
            # Resize to target size if needed
            if (current_h, current_w) != target_size:
                patch_features = torch.nn.functional.interpolate(
                    patch_features, 
                    size=target_size, 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # Flatten back to sequence format for attention
            patch_features = patch_features.permute(0, 2, 3, 1)  # (B, H, W, embed_dim)
            patch_features = patch_features.view(batch_size, -1, embed_dim)  # (B, H*W, embed_dim)
        
        return patch_features


def create_geometry_encoder(config: Dict[str, Any]) -> GeometryEncoder:
    """
    Create a geometry encoder from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        GeometryEncoder instance
    """
    return GeometryEncoder(
        model_name=config.get('model_name', 'dinov2_vitb14'),
        patch_size=config.get('patch_size', 14),
        embed_dim=config.get('embed_dim', 768),
        num_heads=config.get('num_heads', 12),
        num_layers=config.get('num_layers', 12),
        freeze_backbone=config.get('freeze_backbone', True),
        use_cls_token=config.get('use_cls_token', False),
        projection_dim=config.get('projection_dim', None),
    ) 