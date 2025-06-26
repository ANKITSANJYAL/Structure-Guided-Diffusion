import torch
from transformers import CLIPTokenizer, CLIPTextModel


class CLIPTextEncoder:
    def __init__(self, model_name="openai/clip-vit-large-patch14", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name).to(self.device)
        self.text_encoder.eval()

    @torch.no_grad()
    def encode(self, class_labels):
        """
        Args:
            class_labels (List[str]): list of class names like ['daisy', 'sunflower']
        Returns:
            clip_embeddings: Tensor of shape (B, 1, 768)
        """
        inputs = self.tokenizer(class_labels, return_tensors="pt", padding=True).to(self.device)
        outputs = self.text_encoder(**inputs)
        pooled = outputs.last_hidden_state[:, 0, :]  # Use CLS token
        return pooled.unsqueeze(1)  # Shape: (B, 1, 768)
