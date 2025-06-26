import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

class DINOEncoder(nn.Module):
    def __init__(self, model_name="facebook/dinov2-base", device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load DINOv2 model and processor
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def forward(self, images):
        """
        Args:
            images: torch.Tensor or List[PIL.Image] of shape (B, C, H, W) or list of PILs
        Returns:
            dino_tokens: Tensor of shape (B, T, 768), excluding CLS
        """
        if isinstance(images, torch.Tensor):
            # Assume images are already normalized tensors (B, C, H, W)
            inputs = {"pixel_values": images.to(self.device)}
        else:
            # Assume images are list of PIL.Image
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)

        outputs = self.model(**inputs)
        tokens = outputs.last_hidden_state  # (B, T+1, 768)
        return tokens[:, 1:, :]  # Exclude CLS → (B, T, 768)
