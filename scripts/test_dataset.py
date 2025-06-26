#!/usr/bin/env python3
"""
Test script to verify dataset loading and downloading.
"""

import os
import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.dataset import create_dataset
from torchvision import transforms


def test_dataset_loading():
    """Test dataset loading and downloading."""
    
    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "training_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Testing dataset loading...")
    print(f"Dataset: {config['data']['dataset_name']}")
    print(f"Root dir: {config['data']['root_dir']}")
    
    # Create simple transform
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
    
    try:
        # Create dataset with download=True
        print("\nCreating dataset with automatic download...")
        dataset = create_dataset(
            dataset_name=config['data']['dataset_name'],
            root_dir=config['data']['root_dir'],
            split='train',
            transform=transform,
            download=True
        )
        
        print(f"Dataset created successfully!")
        print(f"Number of images: {len(dataset)}")
        
        # Test loading a few samples
        print("\nTesting sample loading...")
        for i in range(min(3, len(dataset))):
            image, label = dataset[i]
            print(f"Sample {i}: Image shape {image.shape}, Label {label}")
        
        # Test validation split
        print("\nTesting validation split...")
        val_dataset = create_dataset(
            dataset_name=config['data']['dataset_name'],
            root_dir=config['data']['root_dir'],
            split='val',
            transform=transform,
            download=False  # Already downloaded
        )
        print(f"Validation dataset: {len(val_dataset)} images")
        
        print("\n✅ Dataset loading test passed!")
        
    except Exception as e:
        print(f"\n❌ Dataset loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = test_dataset_loading()
    sys.exit(0 if success else 1) 