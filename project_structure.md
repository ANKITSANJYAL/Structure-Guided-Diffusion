# Project Structure for GeoDreamer Research

```
Structure-Guided-Diffusion/
├── README.md                          # Research proposal (already exists)
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package setup
├── configs/                          # Configuration files
│   ├── __init__.py
│   ├── training_config.yaml          # Training hyperparameters
│   ├── model_config.yaml             # Model architecture config
│   └── dataset_config.yaml           # Dataset configuration
├── src/                              # Main source code
│   ├── __init__.py
│   ├── models/                       # Model implementations
│   │   ├── __init__.py
│   │   ├── diffusion_model.py        # DDPM with geometry conditioning
│   │   ├── unet.py                   # U-Net architecture
│   │   └── geometry_encoder.py       # DINO-based geometry extraction
│   ├── data/                         # Data handling
│   │   ├── __init__.py
│   │   ├── dataset.py                # Oxford Flowers dataset loader
│   │   ├── transforms.py             # Image transformations
│   │   └── semantic_conditioning.py  # CLIP embedding extraction
│   ├── training/                     # Training utilities
│   │   ├── __init__.py
│   │   ├── trainer.py                # Training loop
│   │   ├── losses.py                 # Loss functions
│   │   └── metrics.py                # Evaluation metrics
│   ├── inference/                    # Inference and sampling
│   │   ├── __init__.py
│   │   ├── sampler.py                # Sampling with geometry guidance
│   │   └── visualization.py          # Result visualization
│   └── utils/                        # Utility functions
│       ├── __init__.py
│       ├── geometry_utils.py         # Geometry processing utilities
│       └── logging_utils.py          # Logging and monitoring
├── notebooks/                        # Jupyter notebooks
│   ├── kaggle_training.ipynb         # Main Kaggle training notebook
│   ├── model_download.ipynb          # Download trained models
│   └── local_inference.ipynb         # Local inference demo
├── scripts/                          # Command-line scripts
│   ├── train.py                      # Training script
│   ├── evaluate.py                   # Evaluation script
│   └── generate.py                   # Generation script
├── data/                             # Data directory
│   ├── raw/                          # Raw dataset files
│   ├── processed/                    # Processed data
│   └── .gitkeep
├── models/                           # Model checkpoints
│   ├── pretrained/                   # Pre-trained models (DINO, CLIP)
│   ├── checkpoints/                  # Training checkpoints
│   └── .gitkeep
├── results/                          # Results and outputs
│   ├── samples/                      # Generated samples
│   ├── evaluations/                  # Evaluation results
│   └── .gitkeep
├── logs/                             # Training logs
│   └── .gitkeep
└── tests/                            # Unit tests
    ├── __init__.py
    ├── test_models.py
    ├── test_data.py
    └── test_training.py
```

## Execution Plan

### Phase 1: Setup and Data Preparation
1. **Environment Setup** (`requirements.txt`, `setup.py`)
2. **Data Pipeline** (`src/data/`)
3. **Configuration Management** (`configs/`)

### Phase 2: Model Architecture
1. **Geometry Encoder** (`src/models/geometry_encoder.py`)
2. **Diffusion Model** (`src/models/diffusion_model.py`)
3. **U-Net with Geometry Conditioning** (`src/models/unet.py`)

### Phase 3: Training Infrastructure
1. **Training Loop** (`src/training/trainer.py`)
2. **Loss Functions** (`src/training/losses.py`)
3. **Metrics and Evaluation** (`src/training/metrics.py`)

### Phase 4: Kaggle Training
1. **Kaggle Notebook** (`notebooks/kaggle_training.ipynb`)
2. **Model Download Script** (`notebooks/model_download.ipynb`)

### Phase 5: Local Inference
1. **Sampling Pipeline** (`src/inference/sampler.py`)
2. **Local Inference Demo** (`notebooks/local_inference.ipynb`)
3. **Command-line Tools** (`scripts/`)

### Phase 6: Evaluation and Analysis
1. **Comprehensive Evaluation** (`scripts/evaluate.py`)
2. **Visualization Tools** (`src/inference/visualization.py`)
3. **Results Analysis** 