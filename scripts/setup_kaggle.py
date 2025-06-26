#!/usr/bin/env python3
"""
Kaggle setup script to handle CUDA version conflicts and install dependencies.
"""

import subprocess
import sys
import os


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}...")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False


def setup_kaggle():
    """Setup environment for Kaggle."""
    print("🚀 Setting up GeoDreamer for Kaggle...")
    
    # Step 1: Uninstall existing torch packages to avoid conflicts
    if not run_command(
        "pip uninstall torch torchvision torchaudio -y",
        "Uninstalling existing PyTorch packages"
    ):
        print("⚠️  Warning: Could not uninstall existing packages, continuing...")
    
    # Step 2: Install compatible PyTorch versions
    if not run_command(
        "pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118 --index-url https://download.pytorch.org/whl/cu118",
        "Installing compatible PyTorch versions"
    ):
        return False
    
    # Step 3: Install other dependencies
    if not run_command(
        "pip install -r requirements.txt",
        "Installing other dependencies"
    ):
        return False
    
    # Step 4: Test imports
    print("\n🧪 Testing imports...")
    test_imports = """
import torch
import torchvision
import transformers
import diffusers
import dinov2
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name()}")
print("✅ All imports successful!")
"""
    
    try:
        exec(test_imports)
        print("✅ Import test passed!")
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False
    
    print("\n🎉 Kaggle setup completed successfully!")
    print("\nNext steps:")
    print("1. Test dataset loading: python scripts/test_dataset.py")
    print("2. Start training: python scripts/train.py --config configs/training_config.yaml")
    
    return True


if __name__ == "__main__":
    success = setup_kaggle()
    sys.exit(0 if success else 1) 