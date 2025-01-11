import os
import yaml
import logging
import torch
from typing import Dict, Any, Optional
from pathlib import Path

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override device if CUDA is not available
    if config['training']['device'] == 'cuda' and not torch.cuda.is_available():
        config['training']['device'] = 'cpu'
        print("CUDA not available, using CPU instead")
    
    return config

def setup_logging(config: Dict[str, Any]) -> None:
    """
    Set up logging configuration
    
    Args:
        config: Configuration dictionary
    """
    logging.basicConfig(
        level=config['logging']['level'],
        format=config['logging']['format'],
        handlers=[
            logging.FileHandler(os.path.join(config['paths']['logs_dir'], 'training.log')),
            logging.StreamHandler()
        ]
    )

def create_directories(config: Dict[str, Any]) -> None:
    """
    Create necessary directories from config
    
    Args:
        config: Configuration dictionary
    """
    for dir_name in config['paths'].values():
        Path(dir_name).mkdir(parents=True, exist_ok=True)

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    accuracy: float,
    config: Dict[str, Any],
    label_mapping: Dict[str, int],
    is_best: bool = False
) -> None:
    """
    Save model checkpoint
    
    Args:
        model: The PyTorch model
        optimizer: The optimizer
        epoch: Current epoch number
        loss: Validation loss
        accuracy: Validation accuracy
        config: Configuration dictionary
        label_mapping: Dictionary mapping labels to indices
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'label_mapping': label_mapping
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(
        config['paths']['checkpoints_dir'],
        f'checkpoint_epoch_{epoch}.pth'
    )
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model if this is the best
    if is_best:
        best_path = os.path.join(
            config['paths']['checkpoints_dir'],
            'best_model.pth'
        )
        torch.save(checkpoint, best_path)

def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Dict[str, Any]:
    """
    Load model checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: The PyTorch model
        optimizer: The optimizer (optional)
        
    Returns:
        Dictionary containing checkpoint information
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint

def get_device(config: Dict[str, Any]) -> torch.device:
    """
    Get the device to use for training
    
    Args:
        config: Configuration dictionary
        
    Returns:
        torch.device object
    """
    return torch.device(config['training']['device'])