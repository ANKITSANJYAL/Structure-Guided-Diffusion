"""
U-Net model with geometry and semantic conditioning for diffusion models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timesteps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class CrossAttention(nn.Module):
    """Cross-attention module for conditioning."""
    
    def __init__(self, query_dim: int, context_dim: int, heads: int = 8, dim_head: int = 64):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        if context is None:
            return x
        
        h = self.heads
        
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        q, k, v = map(lambda t: t.view(*t.shape[:2], h, -1).transpose(1, 2), (q, k, v))
        
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = out.transpose(1, 2).reshape(*x.shape[:2], -1)
        
        return self.to_out(out)


class ResBlock(nn.Module):
    """Residual block with timestep conditioning."""
    
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float = 0.0,
        out_channels: Optional[int] = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )
        
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels),
        )
        
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
        )
        
        if self.out_channels == channels and not use_conv:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)
    
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out.unsqueeze(-1).unsqueeze(-1)
            h = self.out_layers(h)
        
        return self.skip_connection(x) + h


class UNetModel(nn.Module):
    """
    U-Net model with geometry and semantic conditioning.
    
    This model extends the standard U-Net with:
    1. Timestep conditioning
    2. Semantic conditioning via cross-attention
    3. Geometry conditioning via cross-attention
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (8, 16),
        dropout: float = 0.0,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        conv_resample: bool = True,
        num_heads: int = 8,
        use_spatial_transformer: bool = True,
        transformer_depth: int = 1,
        context_dim: Optional[int] = None,
        geometry_dim: Optional[int] = None,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        self.use_spatial_transformer = use_spatial_transformer
        self.transformer_depth = transformer_depth
        self.context_dim = context_dim
        self.geometry_dim = geometry_dim
        
        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Input projection
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        ])
        
        # Downsampling
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                    )
                ]
                ch = mult * model_channels
                
                if ds in attention_resolutions:
                    if use_spatial_transformer:
                        layers.append(
                            SpatialTransformer(
                                ch,
                                num_heads,
                                transformer_depth,
                                context_dim,
                                geometry_dim,
                            )
                        )
                    else:
                        layers.append(
                            AttentionBlock(
                                ch,
                                num_heads=num_heads,
                                num_head_channels=ch // num_heads,
                            )
                        )
                
                self.input_blocks.append(nn.ModuleList(layers))
                input_block_chans.append(ch)
            
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    nn.ModuleList([ResBlock(ch, time_embed_dim, dropout, out_channels=ch)])
                    if conv_resample
                    else nn.ModuleList([nn.AvgPool2d(kernel_size=2, stride=2)])
                )
                input_block_chans.append(ch)
                ds *= 2
        
        # Middle block
        self.middle_block = nn.ModuleList([
            ResBlock(ch, time_embed_dim, dropout),
            SpatialTransformer(
                ch,
                num_heads,
                transformer_depth,
                context_dim,
                geometry_dim,
            ) if use_spatial_transformer else AttentionBlock(ch),
            ResBlock(ch, time_embed_dim, dropout),
        ])
        
        # Upsampling
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                    )
                ]
                ch = model_channels * mult
                
                if ds in attention_resolutions:
                    if use_spatial_transformer:
                        layers.append(
                            SpatialTransformer(
                                ch,
                                num_heads,
                                transformer_depth,
                                context_dim,
                                geometry_dim,
                            )
                        )
                    else:
                        layers.append(
                            AttentionBlock(
                                ch,
                                num_heads=num_heads,
                                num_head_channels=ch // num_heads,
                            )
                        )
                
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch)
                        if conv_resample
                        else nn.Upsample(scale_factor=2, mode="nearest")
                    )
                    ds //= 2
                
                self.output_blocks.append(nn.ModuleList(layers))
        
        # Output projection
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, kernel_size=3, padding=1),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        geometry_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the U-Net.
        
        Args:
            x: Input images
            timesteps: Timestep embeddings
            context: Semantic context (CLIP embeddings)
            geometry_tokens: Geometry tokens (DINO embeddings)
            
        Returns:
            Predicted noise
        """
        # Time embedding
        emb = self.time_embed(timesteps)
        
        # Input projection
        hs = []
        h = x.type(self.input_blocks[0][0].weight.dtype)
        
        # Downsampling
        for module in self.input_blocks:
            if isinstance(module, nn.Conv2d):
                h = module(h)
            else:
                for layer in module:
                    if isinstance(layer, ResBlock):
                        h = layer(h, emb)
                    elif isinstance(layer, (SpatialTransformer, AttentionBlock)):
                        h = layer(h, context, geometry_tokens)
                    else:
                        h = layer(h)
            hs.append(h)
        
        # Middle block
        for module in self.middle_block:
            if isinstance(module, ResBlock):
                h = module(h, emb)
            elif isinstance(module, (SpatialTransformer, AttentionBlock)):
                h = module(h, context, geometry_tokens)
            else:
                h = module(h)
        
        # Upsampling
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in module:
                if isinstance(layer, ResBlock):
                    h = layer(h, emb)
                elif isinstance(layer, (SpatialTransformer, AttentionBlock)):
                    h = layer(h, context, geometry_tokens)
                else:
                    h = layer(h)
        
        # Output projection
        return self.out(h)


class SpatialTransformer(nn.Module):
    """Spatial transformer with cross-attention for conditioning."""
    
    def __init__(
        self,
        channels: int,
        num_heads: int,
        depth: int,
        context_dim: Optional[int] = None,
        geometry_dim: Optional[int] = None,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.depth = depth
        self.context_dim = context_dim
        self.geometry_dim = geometry_dim
        
        # Self-attention
        self.self_attn = CrossAttention(channels, channels, num_heads)
        
        # Cross-attention for semantic conditioning
        if context_dim is not None:
            self.cross_attn = CrossAttention(channels, context_dim, num_heads)
        else:
            self.cross_attn = None
        
        # Cross-attention for geometry conditioning
        if geometry_dim is not None:
            self.geometry_attn = CrossAttention(channels, geometry_dim, num_heads)
        else:
            self.geometry_attn = None
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.norm3 = nn.LayerNorm(channels)
        self.norm4 = nn.LayerNorm(channels)
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        geometry_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the spatial transformer.
        
        Args:
            x: Input features (B, C, H, W)
            context: Semantic context (B, seq_len, context_dim)
            geometry_tokens: Geometry tokens (B, seq_len, geometry_dim)
            
        Returns:
            Transformed features
        """
        # Reshape to sequence format
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).transpose(1, 2)  # (B, H*W, C)
        
        # Self-attention
        x = x + self.self_attn(self.norm1(x))
        
        # Cross-attention for semantic conditioning
        if self.cross_attn is not None and context is not None:
            x = x + self.cross_attn(self.norm2(x), context)
        
        # Cross-attention for geometry conditioning
        if self.geometry_attn is not None and geometry_tokens is not None:
            x = x + self.geometry_attn(self.norm3(x), geometry_tokens)
        
        # Feed-forward network
        x = x + self.ffn(self.norm4(x))
        
        # Reshape back to spatial format
        x = x.transpose(1, 2).view(b, c, h, w)
        
        return x


class AttentionBlock(nn.Module):
    """Standard attention block without cross-attention."""
    
    def __init__(self, channels: int, num_heads: int = 1, num_head_channels: int = -1):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, geometry_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Standard self-attention (ignores context and geometry_tokens)
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        scale = 1 / math.sqrt(math.sqrt(C))
        
        attn = torch.einsum("bchw,bcij->bhwij", q * scale, k * scale)
        attn = attn.softmax(dim=-1)
        
        h = torch.einsum("bhwij,bcij->bchw", attn, v)
        h = self.proj(h)
        
        return x + h 