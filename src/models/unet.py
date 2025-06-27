"""
Baseline U-Net for Structure-Guided-Diffusion Phase 1
Standard DDPM U-Net with CLIP text conditioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ResBlock(nn.Module):
    """Residual block with optional conditioning."""
    
    def __init__(self, channels, emb_channels, dropout=0.1):
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, channels)
        )
        
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        
        if channels != channels:
            self.skip_connection = nn.Conv2d(channels, channels, 1)
        else:
            self.skip_connection = nn.Identity()
    
    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        h = h + emb_out.unsqueeze(-1).unsqueeze(-1)
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class CrossAttention(nn.Module):
    """Cross attention for text conditioning."""
    
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim
        
        self.scale = dim_head ** -0.5
        self.heads = heads
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, context=None):
        context = context if context is not None else x
        
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        q, k, v = map(lambda t: t.reshape(*t.shape[:2], self.heads, -1).transpose(1, 2), (q, k, v))
        
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = out.transpose(1, 2).reshape(*x.shape[:2], -1)
        
        return self.to_out(out)


class UNetModel(nn.Module):
    """Baseline U-Net for diffusion with CLIP conditioning."""
    
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        model_channels=128,
        num_res_blocks=2,
        attention_resolutions=(8, 16),
        dropout=0.1,
        channel_mult=(1, 2, 4, 8),
        num_heads=8,
        use_scale_shift_norm=True,
        resblock_updown=False,
        transformer_depth=1,
        context_dim=768,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        self.use_scale_shift_norm = use_scale_shift_norm
        self.resblock_updown = resblock_updown
        self.transformer_depth = transformer_depth
        self.context_dim = context_dim
        
        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
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
                    ResBlock(ch, time_embed_dim, dropout)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(CrossAttention(ch, context_dim, num_heads))
                self.input_blocks.append(nn.ModuleList(layers))
                input_block_chans.append(ch)
            
            if level != len(channel_mult) - 1:
                self.input_blocks.append(nn.ModuleList([nn.Conv2d(ch, ch, 3, stride=2, padding=1)]))
                input_block_chans.append(ch)
                ds *= 2
        
        # Middle
        self.middle_block = nn.ModuleList([
            ResBlock(ch, time_embed_dim, dropout),
            CrossAttention(ch, context_dim, num_heads),
            ResBlock(ch, time_embed_dim, dropout),
        ])
        
        # Upsampling
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(CrossAttention(ch, context_dim, num_heads))
                if level and i == num_res_blocks:
                    layers.append(nn.ConvTranspose2d(ch, ch, 4, 2, 1))
                    ds //= 2
                self.output_blocks.append(nn.ModuleList(layers))
        
        # Output projection
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )
    
    def forward(self, x, timesteps, context=None):
        """Forward pass with optional text conditioning."""
        # Time embedding
        emb = self.time_embed(timesteps)
        
        # Input
        h = x
        hs = []
        for module in self.input_blocks:
            if isinstance(module, nn.Conv2d):
                h = module(h)
            else:
                for layer in module:
                    if isinstance(layer, ResBlock):
                        h = layer(h, emb)
                    elif isinstance(layer, CrossAttention):
                        h = layer(h, context)
            hs.append(h)
        
        # Middle
        for module in self.middle_block:
            if isinstance(module, ResBlock):
                h = module(h, emb)
            elif isinstance(module, CrossAttention):
                h = module(h, context)
        
        # Output
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in module:
                if isinstance(layer, ResBlock):
                    h = layer(h, emb)
                elif isinstance(layer, CrossAttention):
                    h = layer(h, context)
                elif isinstance(layer, nn.ConvTranspose2d):
                    h = layer(h)
        
        return self.out(h)


def create_unet(config):
    """Factory function to create U-Net from config."""
    unet_config = config['model']['unet']
    return UNetModel(
        in_channels=unet_config['in_channels'],
        out_channels=unet_config['out_channels'],
        model_channels=unet_config['model_channels'],
        num_res_blocks=unet_config['num_res_blocks'],
        attention_resolutions=unet_config['attention_resolutions'],
        dropout=unet_config['dropout'],
        channel_mult=unet_config['channel_mult'],
        num_heads=unet_config['num_heads'],
        use_scale_shift_norm=unet_config['use_scale_shift_norm'],
        resblock_updown=unet_config['resblock_updown'],
        transformer_depth=unet_config['transformer_depth'],
        context_dim=unet_config['context_dim'],
    ) 