"""
Evaluation metrics for HACL Model
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import average_precision_score, roc_auc_score

def compute_iou(predictions, targets, threshold=0.5):
    """
    Compute Intersection over Union (IoU)
    
    Args:
        predictions: (N,) predicted probabilities
        targets: (N,) binary targets
        threshold: probability threshold for binarization
    
    Returns:
        iou: IoU score
    """
    pred_binary = (predictions > threshold).astype(np.int32)
    target_binary = (targets > 0.5).astype(np.int32)
    
    intersection = np.sum(pred_binary & target_binary)
    union = np.sum(pred_binary | target_binary)
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    iou = intersection / union
    return iou

def compute_dice(predictions, targets, threshold=0.5):
    """
    Compute Dice coefficient
    
    Args:
        predictions: (N,) predicted probabilities
        targets: (N,) binary targets
        threshold: probability threshold for binarization
    
    Returns:
        dice: Dice coefficient
    """
    pred_binary = (predictions > threshold).astype(np.int32)
    target_binary = targets.astype(np.int32)
    
    intersection = np.sum(pred_binary & target_binary)
    total = np.sum(pred_binary) + np.sum(target_binary)
    
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    
    dice = 2.0 * intersection / total
    return dice

def compute_precision_recall_f1(predictions, targets, threshold=0.5):
    """
    Compute precision, recall, and F1 score
    
    Args:
        predictions: (N,) predicted probabilities
        targets: (N,) binary targets
        threshold: probability threshold for binarization
    
    Returns:
        precision: Precision score
        recall: Recall score
        f1: F1 score
    """
    pred_binary = (predictions > threshold).astype(np.int32)
    target_binary = targets.astype(np.int32)
    
    precision = precision_score(target_binary, pred_binary, zero_division=0)
    recall = recall_score(target_binary, pred_binary, zero_division=0)
    f1 = f1_score(target_binary, pred_binary, zero_division=0)
    
    return precision, recall, f1

def compute_average_precision(predictions, targets):
    """
    Compute Average Precision (AP)
    
    Args:
        predictions: (N,) predicted probabilities
        targets: (N,) binary targets
    
    Returns:
        ap: Average Precision score
    """
    target_binary = targets.astype(np.int32)
    
    if len(np.unique(target_binary)) < 2:
        return 0.0
    
    ap = average_precision_score(target_binary, predictions)
    return ap

def compute_auc(predictions, targets):
    """
    Compute Area Under the ROC Curve (AUC)
    
    Args:
        predictions: (N,) predicted probabilities
        targets: (N,) binary targets
    
    Returns:
        auc: AUC score
    """
    target_binary = (targets > 0.5).astype(np.int32)
    
    if len(np.unique(target_binary)) < 2:
        return 0.5
    
    auc = roc_auc_score(target_binary, predictions)
    return auc

def compute_aiou(predictions, targets, num_thresholds=20):
    """
    Compute Average IoU (aIoU) across multiple thresholds
    
    Args:
        predictions: (N,) predicted probabilities
        targets: (N,) binary targets
        num_thresholds: number of thresholds to evaluate
    
    Returns:
        aiou: Average IoU score
    """
    thresholds = np.linspace(0, 1, num_thresholds)
    ious = []
    
    for threshold in thresholds:
        iou = compute_iou(predictions, targets, threshold)
        ious.append(iou)
    
    return np.mean(ious)

def compute_sim(predictions, targets):
    """
    Compute Similarity (SIM) - cosine similarity between predictions and targets
    
    Args:
        predictions: (N,) predicted probabilities
        targets: (N,) binary targets
    
    Returns:
        sim: Similarity score
    """
    # Normalize vectors
    pred_norm = predictions / (np.linalg.norm(predictions) + 1e-8)
    target_norm = targets / (np.linalg.norm(targets) + 1e-8)
    
    # Compute cosine similarity
    sim = np.dot(pred_norm, target_norm)
    return sim

def compute_mae(predictions, targets):
    """
    Compute Mean Absolute Error (MAE)
    
    Args:
        predictions: (N,) predicted probabilities
        targets: (N,) binary targets
    
    Returns:
        mae: Mean Absolute Error
    """
    mae = np.mean(np.abs(predictions - targets))
    return mae

def compute_metrics(predictions, targets, threshold=0.5):
    """
    Compute all evaluation metrics
    
    Args:
        predictions: (B, N, 1) or (B*N,) predicted probabilities
        targets: (B, N, 1) or (B*N,) binary targets
        threshold: probability threshold for binarization
    
    Returns:
        metrics: Dictionary of computed metrics
    """
    # Flatten arrays if needed
    if predictions.ndim > 1:
        predictions = predictions.flatten()
    if targets.ndim > 1:
        targets = targets.flatten()
    
    # Remove invalid values
    valid_mask = ~(np.isnan(predictions) | np.isnan(targets))
    predictions = predictions[valid_mask]
    targets = targets[valid_mask]
    
    if len(predictions) == 0:
        return {
            # 'iou': 0.0,
            'dice': 0.0,
            # 'precision': 0.0,
            # 'recall': 0.0,
            # 'f1': 0.0,
            # 'accuracy': 0.0,
            # 'ap': 0.0,
            'auc': 0.5,
            'aiou': 0.0,
            'sim': 0.0,
            'mae': 0.0
        }
    
    # Compute metrics
    # iou = compute_iou(predictions, targets, threshold)
    # dice = compute_dice(predictions, targets, threshold)
    # precision, recall, f1 = compute_precision_recall_f1(predictions, targets, threshold)
    
    # Binary predictions for accuracy
    # pred_binary = (predictions > threshold).astype(np.int32)
    # target_binary = (targets > threshold).astype(np.int32)
    # accuracy = accuracy_score(target_binary, pred_binary)
    
    # Probabilistic metrics
    # ap = compute_average_precision(predictions, targets)
    auc = compute_auc(predictions, targets)
    
    # New required metrics
    aiou = compute_aiou(predictions, targets)
    sim = compute_sim(predictions, targets)
    mae = compute_mae(predictions, targets)
    
    metrics = {
        # 'iou': iou,
        # 'dice': dice,
        # 'precision': precision,
        # 'recall': recall,
        # 'f1': f1,
        # 'accuracy': accuracy,
        # 'ap': ap,
        'auc': auc,
        'aiou': aiou,
        'sim': sim,
        'mae': mae
    }
    
    return metrics

def compute_per_class_metrics(predictions, targets, affordance_ids, num_classes=24):
    """
    Compute metrics for each affordance class
    
    Args:
        predictions: (B, N, 1) predicted probabilities
        targets: (B, N, 1) binary targets
        affordance_ids: (B,) affordance category IDs
        num_classes: number of affordance classes
    
    Returns:
        per_class_metrics: Dictionary of per-class metrics
    """
    per_class_metrics = {}
    
    for class_id in range(num_classes):
        # Get samples for this class
        class_mask = affordance_ids == class_id
        
        if class_mask.sum() == 0:
            continue
        
        class_predictions = predictions[class_mask]
        class_targets = targets[class_mask]
        
        # Compute metrics for this class
        class_metrics = compute_metrics(class_predictions, class_targets)
        per_class_metrics[class_id] = class_metrics
    
    return per_class_metrics

def compute_multi_threshold_metrics(predictions, targets, thresholds=None):
    """
    Compute metrics across multiple thresholds
    
    Args:
        predictions: (N,) predicted probabilities
        targets: (N,) binary targets
        thresholds: List of thresholds to evaluate
    
    Returns:
        multi_threshold_metrics: Dictionary of metrics for each threshold
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)
    
    multi_threshold_metrics = {}
    
    for threshold in thresholds:
        metrics = compute_metrics(predictions, targets, threshold)
        multi_threshold_metrics[threshold] = metrics
    
    return multi_threshold_metrics

def find_best_threshold(predictions, targets, metric='f1'):
    """
    Find the best threshold for a given metric
    
    Args:
        predictions: (N,) predicted probabilities
        targets: (N,) binary targets
        metric: metric to optimize ('f1', 'iou', 'dice', etc.)
    
    Returns:
        best_threshold: optimal threshold
        best_score: best metric score
    """
    thresholds = np.arange(0.1, 1.0, 0.05)
    best_score = 0.0
    best_threshold = 0.5
    
    for threshold in thresholds:
        metrics = compute_metrics(predictions, targets, threshold)
        score = metrics[metric]
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score

class MetricsTracker:
    """
    Class to track metrics during training
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics"""
        self.predictions = []
        self.targets = []
        self.affordance_ids = []
    
    def update(self, predictions, targets, affordance_ids=None):
        """
        Update metrics with new batch
        
        Args:
            predictions: (B, N, 1) predicted probabilities
            targets: (B, N, 1) binary targets
            affordance_ids: (B,) affordance category IDs
        """
        # Convert to numpy if needed
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.cpu().numpy()
        if torch.is_tensor(affordance_ids):
            affordance_ids = affordance_ids.cpu().numpy()
        
        self.predictions.append(predictions)
        self.targets.append(targets)
        
        if affordance_ids is not None:
            self.affordance_ids.append(affordance_ids)
    
    def compute(self):
        """
        Compute final metrics
        
        Returns:
            metrics: Dictionary of computed metrics
        """
        if len(self.predictions) == 0:
            return {}
        
        # Concatenate all predictions and targets
        all_predictions = np.concatenate(self.predictions, axis=0)
        all_targets = np.concatenate(self.targets, axis=0)
        
        # Compute overall metrics
        metrics = compute_metrics(all_predictions, all_targets)
        
        # Compute per-class metrics if affordance IDs are available
        if len(self.affordance_ids) > 0:
            all_affordance_ids = np.concatenate(self.affordance_ids, axis=0)
            # Expand affordance IDs to match point cloud dimensions
            B, N = all_predictions.shape[:2]
            expanded_affordance_ids = np.repeat(all_affordance_ids, N)
            
            per_class_metrics = compute_per_class_metrics(
                all_predictions, all_targets, expanded_affordance_ids
            )
            metrics['per_class'] = per_class_metrics
        
        return metrics