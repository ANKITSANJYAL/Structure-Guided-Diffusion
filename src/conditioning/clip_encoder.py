"""
CLIP Text Encoder for Structure-Guided-Diffusion
Provides semantic conditioning using CLIP's text encoder
"""

import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer


class CLIPTextEncoder(nn.Module):
    """CLIP Text Encoder for semantic conditioning."""
    
    def __init__(self, model_name="openai/clip-vit-base-patch16", projection_dim=768, freeze=True):
        super().__init__()
        
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name)
        self.projection_dim = projection_dim
        
        # Freeze CLIP encoder if requested
        if freeze:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        # Projection layer to match U-Net context dimension
        self.projection = nn.Linear(self.text_encoder.config.hidden_size, projection_dim)
        self.max_length = self.tokenizer.model_max_length
        
    def forward(self, text_prompts):
        """Encode text prompts to semantic embeddings."""
        # Handle both string and list inputs
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        
        # Tokenize
        tokens = self.tokenizer(
            text_prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move to device if available
        if hasattr(self, 'device'):
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
        
        # Encode
        with torch.no_grad() if not self.text_encoder.training else torch.enable_grad():
            text_outputs = self.text_encoder(**tokens)
            
        # Extract CLS token and project
        cls_embeddings = text_outputs.last_hidden_state[:, 0, :]
        projected_embeddings = self.projection(cls_embeddings)
        
        return projected_embeddings
    
    def to(self, device):
        self.device = device
        return super().to(device)


def create_clip_encoder(config):
    """Factory function to create a CLIP text encoder from config."""
    return CLIPTextEncoder(
        model_name=config['model']['clip_encoder']['model_name'],
        projection_dim=config['model']['clip_encoder']['projection_dim'],
        freeze=config['model']['clip_encoder']['freeze']
    )


if __name__ == "__main__":
    # Test the CLIP encoder
    encoder = create_clip_encoder()
    
    # Test with sample prompts
    test_prompts = ["a beautiful daisy", "a red rose in bloom", "a yellow sunflower"]
    
    print("Testing CLIP Text Encoder...")
    embeddings = encoder(test_prompts)
    print(f"Input prompts: {test_prompts}")
    print(f"Output shape: {embeddings.shape}")
    print(f"Output device: {embeddings.device}")
    print("✓ CLIP encoder test completed successfully!") 