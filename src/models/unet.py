from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionWithGeometry(AttnProcessor):
    """
    Custom attention processor to attend over both CLIP and DINO features.
    """

    def __init__(self, original_processor, clip_dim, dino_dim, attn_dim):
        super().__init__()
        self.original_processor = original_processor

        # Project CLIP and DINO to attention key/value dimension
        self.clip_proj = nn.Linear(clip_dim, attn_dim)
        self.dino_proj = nn.Linear(dino_dim, attn_dim)

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        # encoder_hidden_states is expected to be a dict with 'clip' and 'dino'
        clip_cond = encoder_hidden_states["clip"]  # (B, 1, clip_dim)
        dino_tokens = encoder_hidden_states["dino"]  # (B, T, dino_dim)

        # Project both into attention space
        clip_kv = self.clip_proj(clip_cond)  # (B, 1, attn_dim)
        dino_kv = self.dino_proj(dino_tokens)  # (B, T, attn_dim)

        # Concatenate along sequence dimension (T+1)
        combined_kv = torch.cat([clip_kv, dino_kv], dim=1)

        # Now call original processor with custom encoder_hidden_states
        return self.original_processor(
            attn,
            hidden_states,
            encoder_hidden_states=combined_kv,
            attention_mask=None
        )


class GeometryAwareUNet(nn.Module):
    def __init__(self, clip_dim=768, dino_dim=768, model_name="CompVis/stable-diffusion-v1-4"):
        super().__init__()
        self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")

        # Replace attention processors in all cross-attn blocks
        for name, module in self.unet.named_modules():
            if hasattr(module, "set_attn_processor"):
                original_processor = module.attn_processor
                module.set_attn_processor(
                    CrossAttentionWithGeometry(
                        original_processor,
                        clip_dim=clip_dim,
                        dino_dim=dino_dim,
                        attn_dim=original_processor.to_k.weight.shape[-1]
                    )
                )

    def forward(self, x_t, t, clip_embedding, dino_tokens):
        """
        Args:
            x_t: noised image (B, C, H, W)
            t: timestep (B)
            clip_embedding: (B, 1, clip_dim)
            dino_tokens: (B, T, dino_dim)
        Returns:
            predicted noise (B, C, H, W)
        """
        encoder_hidden_states = {
            "clip": clip_embedding,
            "dino": dino_tokens
        }

        return self.unet(
            sample=x_t,
            timestep=t,
            encoder_hidden_states=encoder_hidden_states
        ).sample
