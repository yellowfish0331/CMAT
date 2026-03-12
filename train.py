"""
LAS training script with distributed support.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import create_model, get_loss_function, get_supported_models
from data.piadv2_dataset import get_dataloader as get_piadv2_dataloader
from data.piad_dataset import get_piad_dataloader
from data.laso_dataset import get_laso_dataloader
from utils.metrics import compute_metrics
from utils.utils import save_checkpoint, load_checkpoint, setup_logging

# print("--- Python 环境诊断 ---")
# print(f"Python 解释器路径: {sys.executable}")
# print(f"PyTorch 版本: {torch.__version__}")
# print(f"PyTorch 库文件位置: {torch.__file__}")
# print("------------------------")


def setup_distributed(rank, world_size):
    """Setup distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set the device
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()

def get_distributed_dataloader(config, split='train', rank=0, world_size=1):
    """Create distributed dataloader"""
    # Normalize dataset aliases.
    raw_type = str(config.get('dataset_type', 'piadv2')).lower()
    if raw_type in ('piad',):
        dataset_type = 'piad'
    elif raw_type in ('laso',):
        dataset_type = 'laso'
    elif raw_type in ('piadv2', 'piad_v2', 'piad2'):
        dataset_type = 'piadv2'
    else:
        dataset_type = 'piadv2'
    
    if dataset_type == 'laso':
        from data.laso_dataset import LASODataset, collate_fn
        
        # Map split names for LASO (use the actual requested split)
        laso_split = 'test' if split == 'test' else split
        
        # Create LASO dataset
        dataset = LASODataset(
            run_type=laso_split,
            data_root=config['paths'].get('laso_data_root', None),
            num_points=config['data']['num_points'],
            use_augmentation=(split == 'train'),
            eval_setting='all'
        )
        
        # Create distributed sampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=(split == 'train')
        )
        
        # Create dataloader with custom collate function
        dataloader = DataLoader(
            dataset,
            batch_size=config['training']['batch_size'],
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=(split == 'train'),
            collate_fn=collate_fn
        )
        
        return dataloader, sampler
        
    elif dataset_type == 'piad':
        from data.piad_dataset import PIADDataset
        
        # PIAD数据集路径
        data_root = config['paths']['data_root']
        if split == 'train':
            point_path = os.path.join(data_root, 'Point_Train.txt')
            img_path = os.path.join(data_root, 'Img_Train.txt')
            box_path = os.path.join(data_root, 'Box_Train.txt')
            use_augmentation = True
        else:
            point_path = os.path.join(data_root, 'Point_Test.txt')
            img_path = os.path.join(data_root, 'Img_Test.txt')
            box_path = os.path.join(data_root, 'Box_Test.txt')
            use_augmentation = False
        
        # Create PIAD dataset
        dataset = PIADDataset(
            run_type=split,
            setting_type=config.get('setting_type', 'Seen'),
            point_path=point_path,
            img_path=img_path,
            box_path=box_path,
            image_size=tuple(config['data']['image_size']),
            num_points=config['data']['num_points'],
            use_augmentation=use_augmentation,
            pair_num=config.get('pair_num', 2)
        )
    else:
        # PIADv2 visual-prompt dataset.
        from data.piadv2_dataset import PIADV2Dataset
        
        # 统一路径处理：优先使用小写文件名，兼容大写文件名
        data_root = config['paths']['data_root']
        
        if split == 'train':
            # 优先尝试小写文件名（PIADv2格式）
            point_path = os.path.join(data_root, 'Point_train.txt')
            img_path = os.path.join(data_root, 'Img_train.txt')
            # 如果小写文件不存在，尝试大写文件名（PIAD格式）
            if not os.path.exists(point_path):
                point_path = os.path.join(data_root, 'Point_Train.txt')
            if not os.path.exists(img_path):
                img_path = os.path.join(data_root, 'Img_Train.txt')
            use_augmentation = True
        else:
            # 优先尝试小写文件名（PIADv2格式）
            point_path = os.path.join(data_root, 'Point_test.txt')
            img_path = os.path.join(data_root, 'Img_test.txt')
            # 如果小写文件不存在，尝试大写文件名（PIAD格式）
            if not os.path.exists(point_path):
                point_path = os.path.join(data_root, 'Point_Test.txt')
            if not os.path.exists(img_path):
                img_path = os.path.join(data_root, 'Img_Test.txt')
            use_augmentation = False
        
        # Create dataset
        dataset = PIADV2Dataset(
            run_type=split,
            setting_type='Seen',
            point_path=point_path,
            img_path=img_path,
            image_size=config['data']['image_size'],
            num_points=config['data']['num_points'],
            use_augmentation=use_augmentation
        )
    
    # Create distributed sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=(split == 'train')
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader, sampler

class UnifiedTrainer:
    """
    Unified trainer class supporting both single and distributed training
    """
    
    def __init__(self, config, rank=0, world_size=1):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = world_size > 1
        
        # Setup device
        if self.is_distributed:
            self.device = torch.device(f'cuda:{rank}')
        else:
            # Check hardware config for device preference
            use_cuda = config.get('hardware', {}).get('use_cuda', True)
            if use_cuda and torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        
        if rank == 0:
            print(f"Using device: {self.device}")
            if self.is_distributed:
                print(f"Distributed training on {world_size} GPUs")
        
        # Create unique experiment directory (only on rank 0)
        if rank == 0:
            model_name = self.config['model']['name']
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.exp_name = f"{model_name}_{timestamp}"
            
            self.exp_dir = os.path.join(self.config['paths']['checkpoint_dir'], self.exp_name)
            self.log_dir = os.path.join(self.config['paths']['log_dir'], self.exp_name)
            
            os.makedirs(self.exp_dir, exist_ok=True)
            os.makedirs(self.log_dir, exist_ok=True)
            
            print(f"Experiment directory: {self.exp_dir}")

        # Synchronize experiment path for all processes
        if self.is_distributed:
            # Broadcast exp_dir and log_dir to all processes
            dirs = [self.exp_dir if rank == 0 else None, self.log_dir if rank == 0 else None]
            dist.broadcast_object_list(dirs, src=0)
            if rank != 0:
                self.exp_dir, self.log_dir = dirs
        
        # Setup logging (only on rank 0)
        if rank == 0:
            self.setup_logging()
        
        # Create model
        model_name = config['model']['name']
        if rank == 0:
            print(f"Creating {model_name.upper()} model...")
        self.model = create_model(config).to(self.device)
        
        # Wrap model with DDP if distributed
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[rank], find_unused_parameters=False)
        
        # Setup data loaders
        # Normalize dataset types.
        raw_type = str(config.get('dataset_type', 'piadv2')).lower()
        if raw_type in ('piad',):
            dataset_type = 'piad'
        elif raw_type in ('laso',):
            dataset_type = 'laso'
        elif raw_type in ('piadv2', 'piad_v2', 'piad2'):
            dataset_type = 'piadv2'
        else:
            dataset_type = 'piadv2'

        self.dataset_type = dataset_type
        
        if self.is_distributed:
            self.train_loader, self.train_sampler = get_distributed_dataloader(
                config, split='train', rank=rank, world_size=world_size
            )
            self.val_loader, self.val_sampler = get_distributed_dataloader(
                config, split='test', rank=rank, world_size=world_size
            )
            # For LASO in distributed mode, also prepare seen/unseen evaluation loaders (non-distributed)
            if dataset_type == 'laso':
                self.val_loader_seen = get_laso_dataloader(config, split='test', eval_setting='seen')
                self.val_loader_unseen = get_laso_dataloader(config, split='test', eval_setting='unseen')
        else:
            if dataset_type == 'laso':
                self.train_loader = get_laso_dataloader(config, split='train')
                self.val_loader = get_laso_dataloader(config, split='test')
                # Create additional seen/unseen test loaders for LASO
                self.val_loader_seen = get_laso_dataloader(config, split='test', eval_setting='seen')
                self.val_loader_unseen = get_laso_dataloader(config, split='test', eval_setting='unseen')
            elif dataset_type == 'piad':
                self.train_loader = get_piad_dataloader(config, split='train')
                self.val_loader = get_piad_dataloader(config, split='test')
                # For PIAD, we'll keep the existing seen/unseen structure
                self.val_loader_seen = None
                self.val_loader_unseen = None
            else:  # piadv2
                self.train_loader = get_piadv2_dataloader(config, split='train')
                self.val_loader = get_piadv2_dataloader(config, split='test')
                self.val_loader_seen = None
                self.val_loader_unseen = None
            self.train_sampler = None
            self.val_sampler = None
        
        # Setup optimizer and scheduler
        self.setup_optimizer()
        
        # Setup loss functions
        self.setup_losses()
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_aiou = 0.0
        
        if rank == 0:
            print(f"Model created with {sum(p.numel() for p in self.model.parameters())} parameters")
            print(f"Training dataset size: {len(self.train_loader.dataset)}")
            print(f"Validation dataset size: {len(self.val_loader.dataset)}")
    
    def setup_logging(self):
        """Setup logging and tensorboard"""
        # Create log directory
        # The experiment directory is now created in __init__
        
        # Setup tensorboard
        self.writer = SummaryWriter(self.log_dir)
        
        # Setup file logging
        setup_logging(os.path.join(self.log_dir, 'training.log'))
    
    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        # Base learning rate from config
        base_lr = self.config['training']['learning_rate']
        weight_decay = self.config['training']['weight_decay']

        # Ensure base_lr is a number
        if isinstance(base_lr, (list, tuple)):
            base_lr = base_lr[0]  # Take first value if it's a sequence
        elif isinstance(base_lr, str):
            base_lr = float(base_lr)  # Convert string to float (handles scientific notation)
        base_lr = float(base_lr)

        # Adjust base learning rate for distributed training if applicable
        lr = base_lr * self.world_size if self.is_distributed else base_lr
        
        model_module = self.model.module if self.is_distributed else self.model
        
        # LAS uses different learning rates for the prompt encoder and Point-MAE encoder.
        if self.config['model']['name'] == 'las':
            # Get prompt encoder parameters (either visual or text)
            prompt_params = [p for p in model_module.prompt_encoder.parameters() if p.requires_grad]
            
            # Group 2: Point-MAE parameters (LR * 0.2)
            pointmae_params = list(model_module.point_encoder.parameters())

            # Group 3: Rest of the model parameters (base LR)
            prompt_param_ids = {id(p) for p in prompt_params}
            pointmae_param_ids = {id(p) for p in pointmae_params}
            
            other_params = [
                p for p in model_module.parameters() 
                if id(p) not in prompt_param_ids and id(p) not in pointmae_param_ids
            ]

            # Determine prompt encoder type for naming
            prompt_type = self.config['model'].get('prompt_type', 'visual')
            prompt_name = f'{prompt_type}_encoder'

            param_groups = [
                {'params': prompt_params, 'lr': lr * 0.1, 'name': prompt_name},
                {'params': pointmae_params, 'lr': lr * 0.2, 'name': 'point_encoder'},
                {'params': other_params, 'lr': lr, 'name': 'other_modules'}
            ]

            if self.rank == 0:
                print(f"Optimizer configured with parameter groups for LAS ({prompt_type} prompt):")
                total_params = 0
                for group in param_groups:
                    group_param_count = sum(p.numel() for p in group['params'])
                    total_params += group_param_count
                    print(f"  - Group '{group['name']}': {len(group['params'])} tensors, "
                          f"{group_param_count:,} parameters, lr={group['lr']:.2e}")
                print(f"Total trainable parameters: {total_params:,}")

            optimizer_type = self.config['training']['optimizer']
            if optimizer_type == 'adamw':
                self.optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
            elif optimizer_type == 'adam':
                self.optimizer = optim.Adam(param_groups, weight_decay=weight_decay)
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer_type}")
                
        else:
            # Default optimizer setup for other models
            if self.rank == 0:
                print(f"Using default optimizer for {self.config['model']['name']} model.")
            
            optimizer_type = self.config['training']['optimizer']
            if optimizer_type == 'adamw':
                self.optimizer = optim.AdamW(
                    self.model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay
                )
            elif optimizer_type == 'adam':
                self.optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay
                )
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        # Setup scheduler
        if self.config['training']['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=1e-6
            )
        elif self.config['training']['scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            self.scheduler = None
    
    def setup_losses(self):
        """Setup loss functions"""
        self.loss_function = get_loss_function(self.config)
        self.model_name = self.config['model']['name'].lower()
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        # Set epoch for distributed sampler
        if self.is_distributed and self.train_sampler is not None:
            self.train_sampler.set_epoch(self.epoch)
        
        # Set random seed for reproducible data loading in distributed training
        if self.is_distributed:
            torch.manual_seed(42 + self.epoch + self.rank)
            np.random.seed(42 + self.epoch + self.rank)
        
        total_loss = 0
        seg_loss_total = 0
        cont_loss_total = 0
        
        # Only show progress bar on rank 0
        if self.rank == 0:
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        else:
            progress_bar = self.train_loader
        
        for batch_idx, batch in enumerate(progress_bar):
            # Validate batch data consistency
            try:
                if 'points' in batch:
                    points_shape = batch['points'].shape
                    if len(points_shape) != 3 or points_shape[2] != 3:
                        print(f"Warning: Invalid points shape {points_shape}, skipping batch {batch_idx}")
                        continue
                    if points_shape[1] == 0:
                        print(f"Warning: Empty point cloud in batch {batch_idx}, skipping")
                        continue
                        
                if 'gt_mask' in batch:
                    mask_shape = batch['gt_mask'].shape
                    if 'points' in batch and mask_shape[1] != batch['points'].shape[1]:
                        print(f"Warning: Point-mask mismatch in batch {batch_idx}: {batch['points'].shape[1]} vs {mask_shape[1]}, skipping")
                        continue
                        
            except Exception as e:
                print(f"Error validating batch {batch_idx}: {e}, skipping")
                continue
            
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(batch)
            
            loss, loss_dict = self.loss_function(
                outputs['segmentation_logits'],
                batch['gt_mask']
            )
            seg_loss = loss_dict['focal_loss']
            cont_loss = loss_dict['dice_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            seg_loss_total += seg_loss.item()
            cont_loss_total += cont_loss.item()
            
            # Update progress bar (only on rank 0)
            if self.rank == 0:
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Focal': f'{seg_loss.item():.4f}',
                    'Dice': f'{cont_loss.item():.4f}'
                })
                
                # Log to tensorboard
                global_step = self.epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item(), global_step)
                self.writer.add_scalar('Train/FocalLoss', seg_loss.item(), global_step)
                self.writer.add_scalar('Train/DiceLoss', cont_loss.item(), global_step)
        
        # Epoch averages
        avg_loss = total_loss / len(self.train_loader)
        avg_seg_loss = seg_loss_total / len(self.train_loader)
        avg_cont_loss = cont_loss_total / len(self.train_loader)
        
        return avg_loss, avg_seg_loss, avg_cont_loss
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0
        seg_loss_total = 0
        cont_loss_total = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            # Only show progress bar on rank 0
            if self.rank == 0:
                val_iterator = tqdm(self.val_loader, desc="Validating")
            else:
                val_iterator = self.val_loader
                
            for batch in val_iterator:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch)
                
                loss, loss_dict = self.loss_function(
                    outputs['segmentation_logits'],
                    batch['gt_mask']
                )
                seg_loss = loss_dict['focal_loss']
                cont_loss = loss_dict['dice_loss']
                
                # Update metrics
                total_loss += loss.item()
                seg_loss_total += seg_loss.item()
                cont_loss_total += cont_loss.item()
                
                # Collect predictions for metrics
                predictions = torch.sigmoid(outputs['segmentation_logits']).cpu().numpy()
                targets = batch['gt_mask'].cpu().numpy()
                
                all_predictions.append(predictions)
                all_targets.append(targets)
        
        # Compute metrics
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        metrics = compute_metrics(all_predictions, all_targets)
        
        # Epoch averages
        avg_loss = total_loss / len(self.val_loader)
        avg_seg_loss = seg_loss_total / len(self.val_loader)
        avg_cont_loss = cont_loss_total / len(self.val_loader)
        
        return avg_loss, avg_seg_loss, avg_cont_loss, metrics
    
    def validate_seen_unseen(self):
        """Validate the model on seen/unseen splits for LASO dataset"""
        if self.dataset_type != 'laso':
            return {}
        
        results = {}
        
        # Evaluate on seen split
        if hasattr(self, 'val_loader_seen') and self.val_loader_seen is not None:
            print(f"\n[Rank {self.rank}] Evaluating on SEEN split...")
            seen_results = self._evaluate_split(self.val_loader_seen, "Seen")
            for key, value in seen_results.items():
                results[f'seen_{key}'] = value
        
        # Evaluate on unseen split  
        if hasattr(self, 'val_loader_unseen') and self.val_loader_unseen is not None:
            print(f"\n[Rank {self.rank}] Evaluating on UNSEEN split...")
            unseen_results = self._evaluate_split(self.val_loader_unseen, "Unseen")
            for key, value in unseen_results.items():
                results[f'unseen_{key}'] = value
        
        return results
    
    def _evaluate_split(self, dataloader, split_name):
        """Evaluate model on a specific data split"""
        self.model.eval()
        
        total_loss = 0
        seg_loss_total = 0
        cont_loss_total = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            # Only show progress bar on rank 0
            if self.rank == 0:
                iterator = tqdm(dataloader, desc=f"Evaluating {split_name}")
            else:
                iterator = dataloader
                
            for batch in iterator:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch)
                
                loss, loss_dict = self.loss_function(
                    outputs['segmentation_logits'],
                    batch['gt_mask']
                )
                seg_loss = loss_dict['focal_loss']
                cont_loss = loss_dict['dice_loss']
                
                total_loss += loss.item()
                seg_loss_total += seg_loss.item()
                cont_loss_total += cont_loss.item()
                
                # Collect predictions and targets for metrics
                pred = torch.sigmoid(outputs['segmentation_logits'])
                all_predictions.append(pred.cpu())
                all_targets.append(batch['gt_mask'].cpu())
        
        # Compute metrics (use numpy for consistency with main validate)
        all_predictions = torch.cat(all_predictions, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        
        # Calculate detailed metrics using the same function as main validation
        metrics = compute_metrics(all_predictions, all_targets)
        
        # Epoch averages
        avg_loss = total_loss / len(dataloader)
        avg_seg_loss = seg_loss_total / len(dataloader)
        avg_cont_loss = cont_loss_total / len(dataloader)
        
        return {
            'loss': avg_loss,
            'seg_loss': avg_seg_loss,
            'cont_loss': avg_cont_loss,
            'aiou': metrics['aiou'],
            'auc': metrics['auc'],
            'sim': metrics['sim'],
            'mae': metrics['mae']
        }
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        
        for epoch in range(self.config['training']['epochs']):
            self.epoch = epoch
            
            # Training
            train_loss, train_seg_loss, train_cont_loss = self.train_epoch()
            
            # Validation
            val_loss, val_seg_loss, val_cont_loss, val_metrics = self.validate()
            
            # LASO Seen/Unseen evaluation (every 5 epochs to avoid too much overhead)
            seen_unseen_results = {}
            if self.dataset_type == 'laso':
                seen_unseen_results = self.validate_seen_unseen()
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
            
            # Logging (only on rank 0)
            if self.rank == 0:
                print(f"Epoch {epoch}:")
                print(f"  Train Loss: {train_loss:.4f} (Focal: {train_seg_loss:.4f}, Dice: {train_cont_loss:.4f})")
                print(f"  Val Loss: {val_loss:.4f} (Focal: {val_seg_loss:.4f}, Dice: {val_cont_loss:.4f})")
                print(f"  Val aIoU: {val_metrics['aiou']:.4f}")
                print(f"  Val AUC: {val_metrics['auc']:.4f}")
                print(f"  Val SIM: {val_metrics['sim']:.4f}")
                print(f"  Val MAE: {val_metrics['mae']:.4f}")
                
                # Log seen/unseen results if available
                if seen_unseen_results:
                    print(f"  --- LASO Seen/Unseen Results ---")
                    if 'seen_aiou' in seen_unseen_results:
                        print(
                            f"  Seen aIoU: {seen_unseen_results['seen_aiou']:.4f}, "
                            f"AUC: {seen_unseen_results['seen_auc']:.4f}, "
                            f"SIM: {seen_unseen_results.get('seen_sim', float('nan')):.4f}, "
                            f"MAE: {seen_unseen_results.get('seen_mae', float('nan')):.4f}"
                        )
                    if 'unseen_aiou' in seen_unseen_results:
                        print(
                            f"  Unseen aIoU: {seen_unseen_results['unseen_aiou']:.4f}, "
                            f"AUC: {seen_unseen_results['unseen_auc']:.4f}, "
                            f"SIM: {seen_unseen_results.get('unseen_sim', float('nan')):.4f}, "
                            f"MAE: {seen_unseen_results.get('unseen_mae', float('nan')):.4f}"
                        )
                
                # Tensorboard logging
                self.writer.add_scalar('Train/EpochLoss', train_loss, epoch)
                self.writer.add_scalar('Val/EpochLoss', val_loss, epoch)
                
                self.writer.add_scalar('Train/EpochFocalLoss', train_seg_loss, epoch)
                self.writer.add_scalar('Train/EpochDiceLoss', train_cont_loss, epoch)
                self.writer.add_scalar('Val/EpochFocalLoss', val_seg_loss, epoch)
                self.writer.add_scalar('Val/EpochDiceLoss', val_cont_loss, epoch)
                
                for metric_name, metric_value in val_metrics.items():
                    self.writer.add_scalar(f'Val/{metric_name}', metric_value, epoch)
            
            # Save checkpoint
            is_best = val_metrics['aiou'] > self.best_val_aiou
            if is_best:
                self.best_val_aiou = val_metrics['aiou']
                self.best_val_loss = val_loss
            
            # Use the experiment-specific directory for saving checkpoints
            checkpoint_path = os.path.join(self.exp_dir, 'checkpoint.pth')

            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'best_val_aiou': self.best_val_aiou,
                'best_val_loss': self.best_val_loss,
                'config': self.config
            }, is_best, checkpoint_path)
            
            if self.rank == 0:
                print(f"  Best Val aIoU: {self.best_val_aiou:.4f}")
                print("-" * 50)
        
        print("Training completed!")
        if self.rank == 0:
            self.writer.close()

def train_worker(rank, world_size, config, resume_path=None):
    """Training worker function for distributed training"""
    # Setup distributed training
    setup_distributed(rank, world_size)
    
    try:
        # Create trainer
        trainer = UnifiedTrainer(config, rank=rank, world_size=world_size)
        
        # Resume from checkpoint if specified
        if resume_path:
            checkpoint = load_checkpoint(resume_path)
            
            # Handle DDP state dict
            state_dict = checkpoint['model_state_dict']
            if not list(state_dict.keys())[0].startswith('module.') and world_size > 1:
                # Add 'module.' prefix for DDP
                state_dict = {f'module.{k}': v for k, v in state_dict.items()}
            elif list(state_dict.keys())[0].startswith('module.') and world_size == 1:
                # Remove 'module.' prefix for single GPU
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            
            trainer.model.load_state_dict(state_dict)
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if trainer.scheduler and checkpoint['scheduler_state_dict']:
                trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            trainer.epoch = checkpoint['epoch']
            trainer.best_val_aiou = checkpoint.get('best_val_aiou', checkpoint.get('best_val_iou', 0.0))
            trainer.best_val_loss = checkpoint['best_val_loss']
            
            if rank == 0:
                print(f"Resumed from epoch {trainer.epoch}")
        
        # Start training
        trainer.train()
        
    finally:
        # Clean up distributed training
        cleanup_distributed()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='LAS training script')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--distributed', action='store_true',
                       help='Enable distributed training')
    parser.add_argument('--world-size', type=int, default=1,
                       help='Number of GPUs for distributed training')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate model type
    supported_models = get_supported_models()
    model_name = config['model']['name'].lower()
    if model_name not in supported_models:
        raise ValueError(f"Model '{model_name}' not supported. "
                        f"Supported models: {supported_models}")
    
    print(f"Training {model_name.upper()} model with config: {args.config}")
    
    # Create base directories, experiment-specific ones are created in the trainer
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['paths']['log_dir'], exist_ok=True)
    
    # Check for distributed training
    if args.distributed or args.world_size > 1:
        # Distributed training
        world_size = args.world_size
        print(f"Starting distributed training on {world_size} GPUs")
        
        # Spawn training processes
        mp.spawn(
            train_worker,
            args=(world_size, config, args.resume),
            nprocs=world_size,
            join=True
        )
    else:
        # Single GPU training
        print("Starting single GPU training")
        trainer = UnifiedTrainer(config)
        
        # Resume from checkpoint if specified
        if args.resume:
            checkpoint = load_checkpoint(args.resume)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if trainer.scheduler and checkpoint['scheduler_state_dict']:
                trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            trainer.epoch = checkpoint['epoch']
            trainer.best_val_aiou = checkpoint.get('best_val_aiou', checkpoint.get('best_val_iou', 0.0))
            trainer.best_val_loss = checkpoint['best_val_loss']
            print(f"Resumed from epoch {trainer.epoch}")
        
        # Start training
        trainer.train()

if __name__ == '__main__':
    main()