# GeoDreamer — Geometry-Aware Diffusion for Structured and View-Consistent Image Generation

## Overview

This project proposes **GeoDreamer**, a geometry-aware diffusion model that enhances 2D image generation with spatial coherence, view consistency, and fine-grained structure preservation by incorporating DINO-based geometric tokens and CLIP-driven semantic conditioning. The goal is to bridge the gap between 2D generative quality and 3D structural realism in low-data or few-shot settings.

## Motivation

Despite the impressive progress in diffusion-based generative models (e.g., Stable Diffusion, DALL·E), current methods hallucinate geometry and often produce inconsistent object structures or view-incoherent images. Previous work on NeRF+DINO+LoRA revealed that few-shot 3D reconstruction suffers from spatial blur due to limited views. This project flips that paradigm: instead of reconstructing geometry, we inject learned geometry into 2D generation.

## Methodology

### Dataset

- **Primary Dataset**: Oxford Flowers 102 (~8,000 images at 224×224) for structured, visually diverse, and geometry-rich image generation
- **Optional Extension**: Stanford Cars or CUB-200 for view-consistent analysis

### Semantic Conditioning

- Extract CLIP embeddings of class labels (e.g., "daisy", "sunflower") as `y_clip`

### Geometry Extraction

1. Resize training images to 224×224
2. Use pre-trained DINOv2-ViT to extract frozen patch tokens (excluding CLS)
3. Optionally project or LoRA-tune geometry tokens

### Modified Diffusion Pipeline

- Use a U-Net-based DDPM conditioned on `(x_t, t, y_clip, geometry_tokens)`
- Inject geometry tokens into the cross-attention layers of the U-Net
- Train with standard denoising loss (e.g., L2 on ε prediction)

### Sampling

- Generate images from noise using prompt (CLIP) + geometry prior injection
- Use classifier-free guidance between (text + geometry) vs. null conditions

### Evaluation

- **FID / CLIPSim** for realism and alignment
- **LPIPS / SSIM** (optional) for structural and multi-view consistency
- Visual comparisons against baseline DDPM (no geometry) and LoRA-tuned versions

## Expected Outcomes

- Structurally consistent, prompt-aligned image generation even under low supervision
- Demonstrated improvements over vanilla 2D diffusion in shape coherence and spatial realism
- Optional extensions toward few-view control, geometry-guided editing, or view interpolation

## Resources & Tools

### Computing Infrastructure
- **Training**: Kaggle Notebooks (GPU)
- **Inference, Analysis, and Visualization**: MacBook M4 Pro

### Libraries & Dependencies
- PyTorch
- Transformers
- Taming-transformers
- DINOv2
- CLIP
- Diffusers

## Getting Started

### Prerequisites

1. **Python Environment**: Python 3.8+ with pip
2. **Dependencies**: Install required packages:

   **For Local Development:**
   ```bash
   pip install -r requirements.txt
   ```

   **For Kaggle (to avoid CUDA version conflicts):**
   ```bash
   pip install -r requirements_kaggle.txt
   ```

   **Alternative Kaggle Installation (if issues persist):**
   ```bash
   # First, uninstall existing torch packages
   pip uninstall torch torchvision torchaudio -y
   
   # Install compatible versions
   pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118 --index-url https://download.pytorch.org/whl/cu118
   
   # Install other dependencies
   pip install -r requirements.txt
   ```

### Data Loading

The project automatically handles dataset downloading and processing:

- **Oxford Flowers 102**: Automatically downloaded from the official source when you first run training
- **Dataset Location**: `data/raw/oxford_flowers_102/`
- **Processed Data**: `data/processed/` (automatically created)

### Testing Dataset Loading

Before training, you can test the dataset loading:

```bash
python scripts/test_dataset.py
```

This will:
1. Download the Oxford Flowers 102 dataset (~330MB)
2. Extract and process the images
3. Test loading training and validation splits
4. Verify sample loading works correctly

### Training

1. **Start Training**:
   ```bash
   python scripts/train.py --config configs/training_config.yaml
   ```

2. **Resume Training** (if interrupted):
   ```bash
   python scripts/train.py --config configs/training_config.yaml --resume_from path/to/checkpoint.pth
   ```

### Generation

1. **Generate Images**:
   ```bash
   python scripts/generate.py --config configs/training_config.yaml --checkpoint path/to/model.pth --prompt "a beautiful sunflower"
   ```

### Kaggle Training Workflow

1. **Push to GitHub**: Commit and push all changes to your repository
2. **Clone in Kaggle**: Clone your repo in a Kaggle notebook
3. **Setup Environment**: Run the Kaggle setup script:
   ```bash
   python scripts/setup_kaggle.py
   ```
   This script will:
   - Uninstall conflicting PyTorch versions
   - Install compatible CUDA versions
   - Install all other dependencies
   - Test that everything works correctly
4. **Start Training**: Run the training script directly
5. **Download Results**: Download checkpoints and results to your local machine

**Alternative Manual Setup (if script fails):**
```bash
# Uninstall existing packages
pip uninstall torch torchvision torchaudio -y

# Install compatible versions
pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

The training script automatically:
- Downloads pre-trained models (DINOv2, CLIP) on first run
- Downloads and processes the Oxford Flowers dataset
- Handles all data loading and preprocessing

## Contributing

*[This section will be populated as the project develops]*

## License

*[This section will be populated as the project develops]*

