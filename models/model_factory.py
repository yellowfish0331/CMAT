"""
Model factory for LAS-only training and evaluation.
"""

from .las_model import create_las_model

def create_model(config):
    """
    Create model based on configuration
    
    Args:
        config: Configuration dictionary containing model specifications
    
    Returns:
        model: Model instance
    
    Raises:
        ValueError: If model type is not supported
    """
    model_name = config['model']['name'].lower()
    
    if model_name == 'las':
        return create_las_model(config)

    raise ValueError(f"Unsupported model type: {model_name}. Supported models: ['las']")

def get_loss_function(config):
    """
    Get appropriate loss function based on model type
    
    Args:
        config: Configuration dictionary
    
    Returns:
        loss_fn: Loss function instance
    """
    model_name = config['model']['name'].lower()
    
    if model_name == 'las':
        from .las_model import LASLoss
        return LASLoss(
            focal_alpha=config['loss']['focal_alpha'],
            focal_gamma=config['loss']['focal_gamma'],
            focal_weight=config['loss']['focal_weight'],
            dice_weight=config['loss']['dice_weight']
        )

    raise ValueError(f"Unsupported model type: {model_name}")

def get_supported_models():
    """
    Get list of supported model types
    
    Returns:
        list: List of supported model names
    """
    return ['las']