Proposal: GeoDreamer — Geometry-Aware Diffusion for Structured and View-Consistent Image Generation
Objective
To develop a geometry-aware diffusion model that enhances 2D image generation with spatial coherence, view consistency, and fine-grained structure preservation by incorporating DINO-based geometric tokens and CLIP-driven semantic conditioning. The goal is to bridge the gap between 2D generative quality and 3D structural realism in low-data or few-shot settings.

Motivation
Despite the impressive progress in diffusion-based generative models (e.g., Stable Diffusion, DALL·E), current methods hallucinate geometry and often produce inconsistent object structures or view-incoherent images. Your previous work on NeRF+DINO+LoRA revealed that few-shot 3D reconstruction suffers from spatial blur due to limited views. This project flips that paradigm: instead of reconstructing geometry, we inject learned geometry into 2D generation.

Methodology
Dataset

Start with Oxford Flowers 102 (~8,000 images at 224×224) for structured, visually diverse, and geometry-rich image generation.

Optionally extend to Stanford Cars or CUB-200 for view-consistent analysis.

Semantic Conditioning

Extract CLIP embeddings of class labels (e.g., “daisy”, “sunflower”) as y_clip.

Geometry Extraction

Resize training images to 224×224.

Use pre-trained DINOv2-ViT to extract frozen patch tokens (excluding CLS).

Optionally project or LoRA-tune geometry tokens.

Modified Diffusion Pipeline

Use a U-Net-based DDPM conditioned on (x_t, t, y_clip, geometry_tokens).

Inject geometry tokens into the cross-attention layers of the U-Net.

Train with standard denoising loss (e.g., L2 on ε prediction).

Sampling

Generate images from noise using prompt (CLIP) + geometry prior injection.

Use classifier-free guidance between (text + geometry) vs. null conditions.

Evaluation

FID / CLIPSim for realism and alignment.

LPIPS / SSIM (optional) for structural and multi-view consistency.

Visual comparisons against baseline DDPM (no geometry) and LoRA-tuned versions.

Expected Outcomes
Structurally consistent, prompt-aligned image generation even under low supervision

Demonstrated improvements over vanilla 2D diffusion in shape coherence and spatial realism

Optional extensions toward few-view control, geometry-guided editing, or view interpolation

Resources & Tools
Training on Kaggle Notebooks (GPU)

Inference, analysis, and visualization on MacBook M4 Pro

Libraries: PyTorch, transformers, taming-transformers, DINOv2, CLIP, diffusers

