# GeoDreamer: Geometry-Guided Diffusion via Implicit Spatial Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## 📋 Overview

GeoDreamer is a novel geometry-aware text-to-image diffusion model that learns spatial structure, semantic coherence, and view-consistent features using DINO-based geometric priors and CLIP-driven textual conditioning. The model generates high-quality images from text alone, having implicitly learned spatial priors from geometry supervision during training.

## 🎯 Objective

To develop a diffusion model that addresses the geometric inconsistencies often observed in current text-to-image models like Stable Diffusion, particularly in few-shot or fine-grained domains. The project inverts the traditional paradigm: instead of learning geometry from views, we use geometry as a teacher to improve 2D generation.

## 🚀 Key Features

- **Geometry-Aware Generation**: Implicitly learns spatial structure from DINO-based geometric priors
- **Text-Only Inference**: Generates high-quality images from text prompts without requiring geometric input at inference time
- **Modular Architecture**: Clean separation of CLIP text encoding and DINO geometry encoding
- **Training Flexibility**: Supports modality dropout and classifier-free guidance
- **Comprehensive Evaluation**: Multiple metrics for realism, semantic alignment, and structural consistency

## 🏗️ Architecture

### Core Components

1. **CLIP Text Encoder** (`clip_encoder.py`)
   - Uses CLIPTextModel (ViT-B/16)
   - Extracts [CLS] token embedding for semantic conditioning

2. **DINO Geometry Encoder** (`dino_encoder.py`)
   - Uses frozen DINOv2-ViT-Small
   - Extracts patch tokens (excluding CLS) for geometric conditioning
   - Output shape: `B × T × D`

3. **Geometry-Aware U-Net** (`geometry_aware_unet.py`)
   - U-Net-based DDPM (UNet2DConditionModel)
   - Inputs: `x_t`, `t`, `encoder_hidden_states = concat(clip_proj, dino_proj)`
   - Dimension alignment: `clip_proj (512 → attn_dim)`, `dino_proj (384 → attn_dim)`

### Training Design

- **Modality Dropout**: Randomly drops CLIP or DINO embeddings during training
- **Classifier-Free Guidance (CFG)**: Trains with both condition and null inputs
- **Geometry Supervision**: DINO embeddings provide spatial priors during training only

## 📊 Dataset

### Primary Dataset
- **Oxford Flowers 102**: 8,000+ images across 102 fine-grained flower categories

### Optional Datasets
- **Stanford Cars**: For evaluating view consistency
- **CUB-200**: For shape consistency evaluation

## 📈 Evaluation Metrics

- **FID**: Measures realism vs. dataset distribution
- **CLIPSim**: Evaluates prompt-image semantic alignment
- **LPIPS/SSIM**: Assesses structural and view consistency
- **Human Ranking**: Optional perceptual realism rating

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/ANKITSANJYAL/Structure-Guided-Diffusion.git
cd Structure-Guided-Diffusion

# Install dependencies
pip install -r requirements.txt
```

### Key Dependencies

- PyTorch 2.0+
- Transformers (for CLIP and DINO)
- Diffusers (for diffusion models)
- DINOv2
- CLIP
- NeRFStudio (for NeRF extension)
- Clean-FID
- LPIPS

## 📁 Project Structure

```
geodreamer/
├── data/
│   └── flowers/                 # Oxford Flowers 102 dataset
├── models/
│   ├── clip_encoder.py         # CLIP text encoder
│   ├── dino_encoder.py         # DINO geometry encoder
│   ├── geometry_aware_unet.py  # Main U-Net model
│   └── nerf_teacher.py         # NeRF-based geometry teacher
├── training/
│   ├── train_baseline.py       # Baseline training pipeline
│   ├── train_dropout_cfg.py    # Training with dropout & CFG
│   └── distill_student.py      # Knowledge distillation
├── inference/
│   ├── sample_images.py        # Image generation
│   └── generate_comparisons.py # Comparison visualizations
├── evaluation/
│   ├── eval_metrics.py         # Evaluation metrics
│   └── ablation_plots.ipynb    # Ablation study analysis
└── checkpoints/                # Model checkpoints
```

## 🚀 Usage

### Training

```bash
# Baseline training
python training/train_baseline.py

# Training with modality dropout and CFG
python training/train_dropout_cfg.py

# NeRF-guided training (stretch goal)
python training/train_nerf_guided.py
```

### Inference

```bash
# Generate images from text prompts
python inference/sample_images.py --prompt "a beautiful daisy in bloom"

# Generate comparison images
python inference/generate_comparisons.py
```

### Evaluation

```bash
# Evaluate model performance
python evaluation/eval_metrics.py

# Run ablation studies
python evaluation/ablation_study.py
```

## 🔬 Experimental Phases

1. **Phase 1**: Baseline pipeline implementation
2. **Phase 2**: Conditioning injection and geometry-aware U-Net
3. **Phase 3**: Modality dropout and classifier-free guidance
4. **Phase 4**: Sampling and visualization tools
5. **Phase 5**: Comprehensive metric evaluation
6. **Phase 6**: Ablation studies
7. **Phase 7**: NeRF supervision extension (stretch goal)
8. **Phase 8**: Knowledge distillation (optional)

## 🎯 Expected Outcomes

- Geometry-aware images generated from text-only prompts
- Improved structural consistency and spatial quality compared to baseline DDPM
- Comprehensive ablation studies comparing DINO, CLIP, and combined approaches
- Optional NeRF-based supervision extension

## 🔮 Stretch Goals

### NeRF-Guided Extension

Experiment with NeRF as a geometry teacher:
- Train small NeRF on 10-20 views per class
- Extract scene latent or volume features
- Pass to U-Net as `nerf_proj` tokens
- Train U-Net with CLIP + DINO + NeRF conditioning

## 💻 Development Environment

- **Training**: Kaggle Notebooks (A100/T4 GPU)
- **Inference & Visualization**: MacBook (M4 Pro)
- **Libraries**: PyTorch, Transformers, Diffusers, DINOv2, CLIP, NeRFStudio

## 📝 Research Contributions

This project addresses key limitations in current text-to-image diffusion models:

1. **Geometric Inconsistency**: Current models often hallucinate geometry or exhibit inconsistent object structures
2. **Few-Shot Learning**: Improved performance in fine-grained domains
3. **Spatial Awareness**: Better understanding of spatial relationships and object structure
4. **View Consistency**: More consistent object representations across different viewpoints

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@misc{geodreamer2024,
  title={GeoDreamer: Geometry-Guided Diffusion via Implicit Spatial Learning},
  author={Ankit Sanjyal},
  year={2024},
  url={https://github.com/ANKITSANJYAL/Structure-Guided-Diffusion}
}
```

## 📞 Contact

For questions or collaborations, please reach out to [Your Email] or open an issue on GitHub.

---

**Note**: This is a research project in development. The codebase is structured for rapid iteration and publication readiness, with a modular design that supports various experimental configurations. 