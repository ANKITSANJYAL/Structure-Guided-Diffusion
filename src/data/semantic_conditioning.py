"""
Semantic conditioning using CLIP for text-to-image generation.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Union
import clip
from transformers import CLIPProcessor, CLIPModel


class SemanticConditioner(nn.Module):
    """
    Semantic conditioner using CLIP for text-based conditioning.
    
    This module extracts CLIP embeddings from class labels and provides
    semantic conditioning for the diffusion model.
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        embed_dim: int = 512,
        freeze_backbone: bool = True,
        projection_dim: Optional[int] = None,
    ):
        """
        Initialize the semantic conditioner.
        
        Args:
            model_name: Name of the CLIP model to use
            embed_dim: CLIP embedding dimension
            freeze_backbone: Whether to freeze the CLIP backbone
            projection_dim: Optional projection dimension for output features
        """
        super().__init__()
        
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.projection_dim = projection_dim
        
        # Load CLIP model
        self.clip_model, self.clip_processor = clip.load("ViT-B/32", device="cpu")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.clip_model.parameters():
                param.requires_grad = False
            self.clip_model.eval()
        
        # Optional projection layer
        if projection_dim is not None:
            self.projection = nn.Linear(embed_dim, projection_dim)
            self.output_dim = projection_dim
        else:
            self.projection = nn.Identity()
            self.output_dim = embed_dim
    
    def forward(
        self, 
        text: Union[str, List[str]],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Extract CLIP embeddings from text.
        
        Args:
            text: Input text or list of texts
            normalize: Whether to normalize the embeddings
            
        Returns:
            CLIP embeddings of shape (B, embed_dim)
        """
        if isinstance(text, str):
            text = [text]
        
        # Tokenize text
        text_tokens = clip.tokenize(text).to(self.clip_model.text_projection.device)
        
        # Get text features
        with torch.no_grad() if not self.training else torch.enable_grad():
            text_features = self.clip_model.encode_text(text_tokens)
        
        # Normalize if requested
        if normalize:
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Apply projection if specified
        text_features = self.projection(text_features)
        
        return text_features
    
    def get_feature_dim(self) -> int:
        """Get the output feature dimension."""
        return self.output_dim


class ClassLabelEmbedder:
    """
    Utility class for embedding class labels using CLIP.
    """
    
    def __init__(
        self,
        class_names: List[str],
        semantic_conditioner: SemanticConditioner,
        cache_embeddings: bool = True
    ):
        """
        Initialize the class label embedder.
        
        Args:
            class_names: List of class names
            semantic_conditioner: The semantic conditioner instance
            cache_embeddings: Whether to cache embeddings for efficiency
        """
        self.class_names = class_names
        self.semantic_conditioner = semantic_conditioner
        self.cache_embeddings = cache_embeddings
        
        # Cache for embeddings
        self._embedding_cache = {}
        
        # Pre-compute embeddings if caching is enabled
        if cache_embeddings:
            self._precompute_embeddings()
    
    def _precompute_embeddings(self):
        """Pre-compute embeddings for all class names."""
        print(f"Pre-computing embeddings for {len(self.class_names)} classes...")
        
        for i, class_name in enumerate(self.class_names):
            embedding = self.semantic_conditioner(class_name)
            self._embedding_cache[class_name] = embedding.detach().cpu()
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(self.class_names)} classes")
        
        print("Embedding pre-computation complete!")
    
    def get_embedding(self, class_name: str) -> torch.Tensor:
        """
        Get embedding for a specific class name.
        
        Args:
            class_name: Name of the class
            
        Returns:
            CLIP embedding for the class
        """
        if self.cache_embeddings and class_name in self._embedding_cache:
            return self._embedding_cache[class_name]
        else:
            return self.semantic_conditioner(class_name)
    
    def get_embeddings_batch(self, class_names: List[str]) -> torch.Tensor:
        """
        Get embeddings for a batch of class names.
        
        Args:
            class_names: List of class names
            
        Returns:
            Batch of CLIP embeddings
        """
        embeddings = []
        
        for class_name in class_names:
            embedding = self.get_embedding(class_name)
            embeddings.append(embedding)
        
        return torch.stack(embeddings)
    
    def get_all_embeddings(self) -> torch.Tensor:
        """
        Get embeddings for all class names.
        
        Returns:
            All CLIP embeddings
        """
        return self.get_embeddings_batch(self.class_names)


class OxfordFlowersEmbedder(ClassLabelEmbedder):
    """
    Specialized embedder for Oxford Flowers 102 dataset.
    """
    
    def __init__(self, semantic_conditioner: SemanticConditioner):
        """
        Initialize the Oxford Flowers embedder.
        
        Args:
            semantic_conditioner: The semantic conditioner instance
        """
        # Oxford Flowers 102 class names
        class_names = [
            "pink primrose", "hard-leaved pocket orchid", "canterbury bells",
            "sweet pea", "english marigold", "tiger lily", "moon orchid",
            "bird of paradise", "monkshood", "globe thistle", "snapdragon",
            "colt's foot", "king protea", "spear thistle", "yellow iris",
            "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
            "giant white arum lily", "fire lily", "pincushion flower",
            "fritillary", "red ginger", "grape hyacinth", "corn poppy",
            "prince of wales feathers", "stemless gentian", "artichoke",
            "sweet william", "carnation", "garden phlox", "love in the mist",
            "mexican aster", "alpine sea holly", "ruby-lipped cattleya",
            "cape flower", "great masterwort", "siam tulip", "lenten rose",
            "barbeton daisy", "daffodil", "sword lily", "poinsettia",
            "bolero deep blue", "wallflower", "marigold", "buttercup",
            "oxeye daisy", "common dandelion", "petunia", "wild pansy",
            "primula", "sunflower", "pelargonium", "bishop of llandaff",
            "gaura", "geranium", "orange dahlia", "pink-yellow dahlia",
            "cautleya spicata", "japanese anemone", "black-eyed susan",
            "silverbush", "californian poppy", "osteospermum", "spring crocus",
            "bearded iris", "windflower", "tree poppy", "gazania",
            "azalea", "water lily", "rose", "thorn apple", "morning glory",
            "passion flower", "lotus", "toad lily", "anthurium", "frangipani",
            "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow",
            "magnolia", "cyclamen", "watercress", "canna lily", "hippeastrum",
            "bee balm", "ball moss", "foxglove", "bougainvillea", "camellia",
            "mallow", "mexican petunia", "bromelia", "blanket flower",
            "trumpet creeper", "blackberry lily", "common tulip", "wild rose"
        ]
        
        super().__init__(class_names, semantic_conditioner)


def create_semantic_conditioner(config: Dict[str, Any]) -> SemanticConditioner:
    """
    Create a semantic conditioner from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        SemanticConditioner instance
    """
    return SemanticConditioner(
        model_name=config.get('model_name', 'openai/clip-vit-base-patch32'),
        embed_dim=config.get('embed_dim', 512),
        freeze_backbone=config.get('freeze_backbone', True),
        projection_dim=config.get('projection_dim', None),
    ) 