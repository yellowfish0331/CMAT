"""
Utility functions for HACL Model
"""

import os
import logging
import torch
import numpy as np
import random
import json
from datetime import datetime
import shutil

def set_seed(seed=42):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(log_file=None, level=logging.INFO):
    """
    Setup logging configuration
    
    Args:
        log_file: Path to log file
        level: Logging level
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )

def save_checkpoint(state, is_best, checkpoint_path, best_path=None):
    """
    Save model checkpoint
    
    Args:
        state: State dictionary to save
        is_best: Whether this is the best model so far
        checkpoint_path: Path to save checkpoint
        best_path: Path to save best model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # Save checkpoint
    torch.save(state, checkpoint_path)
    
    # Save best model if this is the best
    if is_best:
        if best_path is None:
            best_path = checkpoint_path.replace('.pth', '_best.pth')
        shutil.copyfile(checkpoint_path, best_path)
        logging.info(f"New best model saved to {best_path}")

def load_checkpoint(checkpoint_path, map_location=None):
    """
    Load model checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint
        map_location: Device to load checkpoint on
    
    Returns:
        state: Loaded state dictionary
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    state = torch.load(checkpoint_path, map_location=map_location)
    logging.info(f"Checkpoint loaded from {checkpoint_path}")
    
    return state

def save_config(config, save_path):
    """
    Save configuration to JSON file
    
    Args:
        config: Configuration dictionary
        save_path: Path to save configuration
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)

def load_config(config_path):
    """
    Load configuration from JSON file
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        config: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

def create_experiment_dir(base_dir, experiment_name=None):
    """
    Create experiment directory with timestamp
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Optional experiment name
    
    Returns:
        exp_dir: Created experiment directory
    """
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    exp_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'results'), exist_ok=True)
    
    return exp_dir

def count_parameters(model):
    """
    Count total and trainable parameters in model
    
    Args:
        model: PyTorch model
    
    Returns:
        total_params: Total number of parameters
        trainable_params: Number of trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params

def get_model_size(model):
    """
    Get model size in MB
    
    Args:
        model: PyTorch model
    
    Returns:
        size_mb: Model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

def format_time(seconds):
    """
    Format time in seconds to human readable format
    
    Args:
        seconds: Time in seconds
    
    Returns:
        formatted_time: Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

def print_model_info(model, input_shape=None):
    """
    Print model information
    
    Args:
        model: PyTorch model
        input_shape: Input shape for model
    """
    total_params, trainable_params = count_parameters(model)
    model_size = get_model_size(model)
    
    print("="*50)
    print("MODEL INFORMATION")
    print("="*50)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {model_size:.2f} MB")
    
    if input_shape is not None:
        print(f"Input shape: {input_shape}")
    
    print("="*50)

def create_dirs_if_not_exist(paths):
    """
    Create directories if they don't exist
    
    Args:
        paths: List of paths to create
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)

def backup_code(source_dir, backup_dir):
    """
    Backup source code
    
    Args:
        source_dir: Source directory to backup
        backup_dir: Backup directory
    """
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    
    shutil.copytree(source_dir, backup_dir, 
                    ignore=shutil.ignore_patterns('*.pyc', '__pycache__', '*.pth', 'logs', 'results'))

def check_cuda_memory():
    """
    Check CUDA memory usage
    
    Returns:
        memory_info: Dictionary with memory information
    """
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    
    memory_info = {
        "cuda_available": True,
        "num_gpus": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "device_name": torch.cuda.get_device_name(),
        "memory_allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
        "memory_reserved": torch.cuda.memory_reserved() / 1024**3,    # GB
        "memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
    }
    
    return memory_info

def print_cuda_info():
    """Print CUDA information"""
    memory_info = check_cuda_memory()
    
    print("="*50)
    print("CUDA INFORMATION")
    print("="*50)
    
    if memory_info["cuda_available"]:
        print(f"CUDA available: Yes")
        print(f"Number of GPUs: {memory_info['num_gpus']}")
        print(f"Current device: {memory_info['current_device']}")
        print(f"Device name: {memory_info['device_name']}")
        print(f"Memory allocated: {memory_info['memory_allocated']:.2f} GB")
        print(f"Memory reserved: {memory_info['memory_reserved']:.2f} GB")
        print(f"Total memory: {memory_info['memory_total']:.2f} GB")
    else:
        print("CUDA available: No")
    
    print("="*50)

class AverageMeter:
    """
    Computes and stores the average and current value
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all values"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        """Update with new value"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class EarlyStopping:
    """
    Early stopping utility
    """
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss, model):
        """
        Check if training should stop
        
        Args:
            val_loss: Current validation loss
            model: Model to save weights
        
        Returns:
            should_stop: Whether to stop training
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        
        return False
    
    def save_checkpoint(self, model):
        """Save model checkpoint"""
        self.best_weights = model.state_dict().copy()

def ensure_dir_exists(path):
    """
    Ensure directory exists, create if not
    
    Args:
        path: Directory path
    """
    os.makedirs(path, exist_ok=True)

def get_git_hash():
    """
    Get current git commit hash
    
    Returns:
        git_hash: Current git commit hash
    """
    try:
        import subprocess
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        return git_hash
    except:
        return "unknown"