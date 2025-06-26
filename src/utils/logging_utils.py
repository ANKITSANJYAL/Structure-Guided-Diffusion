"""
Logging utilities for GeoDreamer training and inference.
"""

import os
import logging
from typing import Optional, Dict, Any
import wandb


def setup_logging(output_dir: str, level: int = logging.INFO) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        output_dir: Output directory for log files
        level: Logging level
        
    Returns:
        Configured logger
    """
    # Create logs directory
    logs_dir = os.path.join(output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('geodreamer')
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = os.path.join(logs_dir, 'training.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def setup_wandb(
    project: str = 'geodreamer',
    run_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[wandb.run]:
    """
    Setup WandB logging.
    
    Args:
        project: WandB project name
        run_name: WandB run name
        config: Configuration to log
        
    Returns:
        WandB run object or None if WandB is not available
    """
    try:
        # Initialize WandB
        wandb_run = wandb.init(
            project=project,
            name=run_name,
            config=config,
        )
        
        print(f"WandB initialized: {wandb_run.url}")
        return wandb_run
        
    except Exception as e:
        print(f"Failed to initialize WandB: {e}")
        print("Continuing without WandB logging")
        return None


def log_metrics(
    wandb_run: Optional[wandb.run],
    metrics: Dict[str, float],
    step: Optional[int] = None,
):
    """
    Log metrics to WandB.
    
    Args:
        wandb_run: WandB run object
        metrics: Dictionary of metrics to log
        step: Step number for logging
    """
    if wandb_run is not None:
        if step is not None:
            wandb_run.log(metrics, step=step)
        else:
            wandb_run.log(metrics)


def log_images(
    wandb_run: Optional[wandb.run],
    images: Dict[str, Any],
    step: Optional[int] = None,
):
    """
    Log images to WandB.
    
    Args:
        wandb_run: WandB run object
        images: Dictionary of images to log
        step: Step number for logging
    """
    if wandb_run is not None:
        if step is not None:
            wandb_run.log(images, step=step)
        else:
            wandb_run.log(images)


def create_experiment_name(config: Dict[str, Any]) -> str:
    """
    Create a descriptive experiment name from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Experiment name
    """
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    
    # Extract key parameters
    dataset = config.get('data', {}).get('dataset_name', 'unknown')
    model_channels = model_config.get('unet', {}).get('model_channels', 128)
    lr = training_config.get('learning_rate', 1e-4)
    epochs = training_config.get('epochs', 100)
    
    # Create name
    name = f"{dataset}_ch{model_channels}_lr{lr}_ep{epochs}"
    
    return name


def save_config(config: Dict[str, Any], output_dir: str, filename: str = 'config.yaml'):
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        output_dir: Output directory
        filename: Configuration filename
    """
    import yaml
    
    config_path = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Configuration saved to {config_path}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config 