#!/usr/bin/env python3
"""
Download pre-trained models for GeoDreamer inference.

This script downloads the required pre-trained models:
- DINOv2 for geometry feature extraction
- CLIP for semantic conditioning
"""

import os
import sys
import torch
import clip
import dinov2

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.geometry_encoder import GeometryEncoder
from data.semantic_conditioning import SemanticConditioner


def download_dinov2():
    """Download and test DINOv2 model."""
    print("Downloading DINOv2 model...")
    
    try:
        # This will automatically download the model on first use
        geometry_encoder = GeometryEncoder(
            model_name="dinov2_vitb14",
            freeze_backbone=True
        )
        
        # Test with a dummy image
        test_image = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = geometry_encoder(test_image)
        
        print("✅ DINOv2 model downloaded and tested successfully!")
        print(f"   Feature dimension: {geometry_encoder.get_feature_dim()}")
        print(f"   Patch size: {geometry_encoder.get_patch_size()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to download DINOv2 model: {e}")
        return False


def download_clip():
    """Download and test CLIP model."""
    print("Downloading CLIP model...")
    
    try:
        # This will automatically download the model on first use
        semantic_conditioner = SemanticConditioner(
            model_name="openai/clip-vit-base-patch32",
            freeze_backbone=True
        )
        
        # Test with a dummy text
        test_text = ["daisy", "rose"]
        with torch.no_grad():
            embeddings = semantic_conditioner(test_text)
        
        print("✅ CLIP model downloaded and tested successfully!")
        print(f"   Embedding dimension: {semantic_conditioner.get_feature_dim()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to download CLIP model: {e}")
        return False


def main():
    """Download all required models."""
    print("=" * 60)
    print("GeoDreamer - Model Download Script")
    print("=" * 60)
    
    # Create models directory
    os.makedirs("models/pretrained", exist_ok=True)
    
    # Download DINOv2
    dinov2_success = download_dinov2()
    
    print()
    
    # Download CLIP
    clip_success = download_clip()
    
    print()
    print("=" * 60)
    
    if dinov2_success and clip_success:
        print("🎉 All models downloaded successfully!")
        print("You can now run inference with:")
        print("  python scripts/generate.py --checkpoint models/checkpoints/best_model.pt --class_idx 5")
    else:
        print("⚠️  Some models failed to download.")
        print("Please check your internet connection and try again.")
    
    print("=" * 60)


if __name__ == "__main__":
    main() 