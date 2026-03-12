"""
Loss functions for HACL Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SegmentationLoss(nn.Module):
    """
    Combined segmentation loss using Focal Loss + Dice Loss
    """
    
    def __init__(self, 
                 focal_alpha=0.25,
                 focal_gamma=2.0,
                 dice_smooth=1e-5,
                 focal_weight=0.5,
                 dice_weight=0.5):
        super().__init__()
        
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.dice_smooth = dice_smooth
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (B, N, 1) logits
            targets: (B, N, 1) binary targets
        
        Returns:
            loss: scalar tensor
        """
        # Focal Loss
        focal_loss = self.focal_loss(predictions, targets)
        
        # Dice Loss
        dice_loss = self.dice_loss(predictions, targets)
        
        # Combined loss
        total_loss = self.focal_weight * focal_loss + self.dice_weight * dice_loss
        
        return total_loss
    
    def focal_loss(self, predictions, targets, focal_alpha=0.25, focal_gamma=2.0):
        """
        Focal Loss implementation (modified to be similar to HM_Loss's structure)
        
        Args:
            predictions: (B, N, 1) logits (will be sigmoid activated internally)
            targets: (B, N, 1) binary targets (0 or 1)
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
        
        Returns:
            loss: scalar tensor
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(predictions)
        
        # Flatten for easier computation
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        # Add a small epsilon to probabilities to prevent log(0)
        epsilon = 1e-6
        probs_flat = torch.clamp(probs_flat, epsilon, 1.0 - epsilon)

        # Calculate loss for positive samples (target=1)
        # Term for true positives: -alpha * (1-p)^gamma * log(p) * target
        term_pos = -focal_alpha * ((1 - probs_flat) ** focal_gamma) * \
                torch.log(probs_flat) * targets_flat

        # Calculate loss for negative samples (target=0)
        # Term for true negatives: -(1-alpha) * p^gamma * log(1-p) * (1-target)
        term_neg = -(1 - focal_alpha) * (probs_flat ** focal_gamma) * \
                torch.log(1 - probs_flat) * (1 - targets_flat)

        # Sum the terms for all pixels and take the mean
        focal_loss = torch.mean(term_pos + term_neg)
        
        return focal_loss
    
    def dice_loss(self, predictions, targets, dice_smooth=1e-6):
        """
        Dice Loss implementation (modified to be similar to HM_Loss's structure - bi-directional)
        
        Args:
            predictions: (B, N, 1) logits (will be sigmoid activated internally)
            targets: (B, N, 1) binary targets (0 or 1)
            dice_smooth: Smoothing factor to prevent division by zero
        
        Returns:
            loss: scalar tensor
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(predictions)
        
        # Flatten for easier computation
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        # --- Positive (Foreground) Dice Calculation ---
        intersection_positive = (probs_flat * targets_flat).sum()
        cardinality_positive = (probs_flat + targets_flat).sum() 
        
        # Compute dice coefficient for positive class
        dice_positive = (2.0 * intersection_positive + dice_smooth) / \
                        (cardinality_positive + dice_smooth)

        # --- Negative (Background) Dice Calculation ---
        # Intersection for background: (1-pred) * (1-target)
        intersection_negative = ((1.0 - probs_flat) * (1.0 - targets_flat)).sum()
        # Cardinality for background: (1-pred) + (1-target)
        cardinality_negative = ((1.0 - probs_flat) + (1.0 - targets_flat)).sum()

        # Compute dice coefficient for negative class
        dice_negative = (2.0 * intersection_negative + dice_smooth) / \
                        (cardinality_negative + dice_smooth)

        # --- Modified Combine Dice coefficients for proper loss behavior ---
        # A common way to combine bi-directional dice loss is to sum the (1 - dice) for each class,
        # or take their average. This ensures loss is 0 for perfect prediction and non-negative.
        
        loss_positive = 1.0 - dice_positive
        loss_negative = 1.0 - dice_negative

        # Option 1: Simple Sum (Common)
        dice_loss = loss_positive + loss_negative 
        
        # Option 2: Averaged (Also Common, especially for multi-class)
        # dice_loss = (loss_positive + loss_negative) / 2.0 
        
        # Option 3: Weighted Sum (if one class is more important or imbalanced)
        # alpha_pos = 0.5 # Example weight for positive class
        # alpha_neg = 0.5 # Example weight for negative class
        # dice_loss = alpha_pos * loss_positive + alpha_neg * loss_negative

        return dice_loss

class ContrastiveLoss(nn.Module):
    """
    InfoNCE Contrastive Loss for affordance learning
    """
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, projections, affordance_ids, instance_ids):
        """
        Compute InfoNCE contrastive loss
        
        Args:
            projections: (B, proj_dim) normalized projections
            affordance_ids: (B,) affordance category IDs
            instance_ids: (B,) instance IDs
        
        Returns:
            loss: scalar tensor
        """
        B = projections.shape[0]
        
        if B < 2:
            return torch.tensor(0.0, device=projections.device)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(projections, projections.T) / self.temperature
        
        # Create masks
        affordance_mask = affordance_ids.unsqueeze(1) == affordance_ids.unsqueeze(0)  # (B, B)
        instance_mask = instance_ids.unsqueeze(1) == instance_ids.unsqueeze(0)  # (B, B)
        
        # Positive pairs: same affordance, different instance
        positive_mask = affordance_mask & ~instance_mask
        
        # Negative pairs: different affordance
        negative_mask = ~affordance_mask
        
        # Self-mask (diagonal)
        self_mask = torch.eye(B, device=projections.device).bool()
        
        # Compute InfoNCE loss
        total_loss = 0
        num_valid_anchors = 0
        
        for i in range(B):
            # Get positive and negative indices for anchor i
            pos_indices = positive_mask[i].nonzero(as_tuple=True)[0]
            neg_indices = negative_mask[i].nonzero(as_tuple=True)[0]
            
            if len(pos_indices) > 0:
                # Get all non-self indices for denominator
                all_indices = (~self_mask[i]).nonzero(as_tuple=True)[0]
                
                # Compute numerator (positive pairs)
                pos_logits = similarity_matrix[i][pos_indices]
                
                # Compute denominator (all pairs except self)
                all_logits = similarity_matrix[i][all_indices]
                
                # Compute log-sum-exp for stability
                max_logit = torch.max(all_logits)
                exp_logits = torch.exp(all_logits - max_logit)
                log_sum_exp = torch.log(exp_logits.sum()) + max_logit
                
                # Compute loss for each positive pair
                for pos_logit in pos_logits:
                    loss = -pos_logit + log_sum_exp
                    total_loss += loss
                    num_valid_anchors += 1
        
        if num_valid_anchors > 0:
            return total_loss / num_valid_anchors
        else:
            return torch.tensor(0.0, device=projections.device)

class AdversarialContrastiveLoss(nn.Module):
    """
    Adversarial contrastive loss with hard negative mining
    """
    
    def __init__(self, temperature=0.07, margin=0.2):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, projections, affordance_ids, instance_ids):
        """
        Compute adversarial contrastive loss
        
        Args:
            projections: (B, proj_dim) normalized projections
            affordance_ids: (B,) affordance category IDs
            instance_ids: (B,) instance IDs
        
        Returns:
            loss: scalar tensor
        """
        B = projections.shape[0]
        
        if B < 2:
            return torch.tensor(0.0, device=projections.device)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(projections, projections.T)
        
        # Create masks
        affordance_mask = affordance_ids.unsqueeze(1) == affordance_ids.unsqueeze(0)
        instance_mask = instance_ids.unsqueeze(1) == instance_ids.unsqueeze(0)
        
        # Positive pairs: same affordance, different instance
        positive_mask = affordance_mask & ~instance_mask
        
        # Negative pairs: different affordance
        negative_mask = ~affordance_mask
        
        # Self-mask
        self_mask = torch.eye(B, device=projections.device).bool()
        
        total_loss = 0
        num_pairs = 0
        
        for i in range(B):
            pos_indices = positive_mask[i].nonzero(as_tuple=True)[0]
            neg_indices = negative_mask[i].nonzero(as_tuple=True)[0]
            
            if len(pos_indices) > 0 and len(neg_indices) > 0:
                # Get positive similarities
                pos_similarities = similarity_matrix[i][pos_indices]
                
                # Get negative similarities
                neg_similarities = similarity_matrix[i][neg_indices]
                
                # Hard negative mining: select hardest negatives
                hard_neg_similarities = neg_similarities.max()
                
                # Compute loss for each positive pair
                for pos_sim in pos_similarities:
                    # Triplet-like loss
                    loss = torch.clamp(hard_neg_similarities - pos_sim + self.margin, min=0)
                    total_loss += loss
                    num_pairs += 1
        
        if num_pairs > 0:
            return total_loss / num_pairs
        else:
            return torch.tensor(0.0, device=projections.device)

class CombinedLoss(nn.Module):
    """
    Combined loss function for HACL model
    """
    
    def __init__(self, 
                 seg_weight=1.0,
                 cont_weight=0.5,
                 temperature=0.07):
        super().__init__()
        
        self.seg_weight = seg_weight
        self.cont_weight = cont_weight
        
        self.segmentation_loss = SegmentationLoss()
        self.contrastive_loss = ContrastiveLoss(temperature=temperature)
    
    def forward(self, outputs, batch):
        """
        Compute combined loss
        
        Args:
            outputs: Dictionary with model outputs
            batch: Dictionary with batch data
        
        Returns:
            loss_dict: Dictionary with individual and total losses
        """
        # Segmentation loss
        seg_loss = self.segmentation_loss(
            outputs['segmentation_logits'],
            batch['gt_mask']
        )
        
        # Contrastive loss
        cont_loss = self.contrastive_loss(
            outputs['contrastive_projections'],
            batch['affordance_id'],
            batch['instance_id']
        )
        
        # Total loss
        total_loss = self.seg_weight * seg_loss + self.cont_weight * cont_loss
        
        return {
            'total_loss': total_loss,
            'segmentation_loss': seg_loss,
            'contrastive_loss': cont_loss
        }