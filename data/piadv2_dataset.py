"""
PIADv2 dataset implementation for LAS visual-prompt training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import json
import random
import os

def pc_normalize(pc):
    """Normalize point cloud to unit sphere"""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, centroid, m

class PIADV2Dataset(Dataset):
    """
    Dataset class for LAS training on PIADv2
    Returns a dictionary containing:
    - 'image': preprocessed image tensor
    - 'points': original point cloud tensor (N x 3)
    - 'gt_mask': pixel-level ground truth mask (N x 1)
    - 'affordance_id': affordance category ID (0-23)
    - 'instance_id': unique ID for different 3D models
    """
    
    def __init__(self, 
                 run_type='train', 
                 setting_type='Seen',
                 point_path=None,
                 img_path=None,
                 image_size=(224, 224),
                 num_points=2048,
                 use_augmentation=True):
        
        super().__init__()
        
        self.run_type = run_type
        self.setting_type = setting_type
        self.image_size = image_size
        self.num_points = num_points
        self.use_augmentation = use_augmentation
        
        # PIADv2 affordance categories (24 classes)
        self.affordance_label_list = [
            'grasp', 'contain', 'lift', 'open', 'lay', 'sit', 'support', 
            'wrapgrasp', 'pour', 'move', 'display', 'push', 'listen', 
            'wear', 'press', 'cut', 'stab', 'carry', 'ride', 'clean', 
            'play', 'beat', 'speak', 'pull'
        ]
        
        # Load file paths
        self.img_files = self._read_file_list(img_path)
        self.point_files = self._read_file_list(point_path)

        # Debug: print a few resolved paths to help diagnose path issues
        if len(self.img_files) > 0:
            print("[PIADV2Dataset] Sample image path resolved:", self.img_files[0])
        if len(self.point_files) > 0:
            print("[PIADV2Dataset] Sample point path resolved:", self.point_files[0])
        
        if self.run_type == 'train':
            self.object_point_map = self._create_object_point_map()
        
        # Create instance mappings
        self.instance_mapping = self._create_instance_mapping()
        
        # Image preprocessing
        self.image_transform = self._get_image_transform()
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index):
        """
        Returns a dictionary with keys:
        - 'image': preprocessed image tensor
        - 'points': point cloud tensor (N x 3)
        - 'gt_mask': ground truth mask (N x 1)
        - 'affordance_id': affordance category ID
        - 'instance_id': unique instance ID
        """
        img_path = self.img_files[index]
        
        if self.run_type == 'train':
            # Dynamic sampling for training to ensure correct image-point cloud pairs.
            # This logic is inspired by the PIAD dataset to prevent mismatches.
            object_name = img_path.split('/')[-4]
            img_affordance_name = img_path.split('/')[-2]
            
            candidate_indices = self.object_point_map.get(object_name)
            if not candidate_indices:
                # Handle cases where an object in the image list has no corresponding point cloud
                raise ValueError(f"No point clouds found for object category: {object_name}")

            while True:
                point_idx = random.choice(candidate_indices)
                point_path = self.point_files[point_idx]
                pc_affordance_name = point_path.split('/')[-2]
                if img_affordance_name == pc_affordance_name:
                    break
        else:
            # Static mapping for validation/testing, assuming files are perfectly aligned
            point_path = self.point_files[index]
        
        # Load and preprocess image
        image = Image.open(img_path).convert('RGB')
        image = image.resize(self.image_size)
        image = self.image_transform(image)
        
        # Load and preprocess point cloud
        points, gt_mask = self._load_point_cloud(point_path)
        points = self._preprocess_points(points)
        
        # Get affordance and instance IDs
        affordance_id = self._get_affordance_id(img_path)
        instance_id = self._get_instance_id(img_path)
        
        return {
            'image': image,
            'points': torch.from_numpy(points).float(),
            'gt_mask': torch.from_numpy(gt_mask).float(),
            'affordance_id': affordance_id,
            'instance_id': instance_id
        }
    
    def _read_file_list(self, path):
        """Read file list from text file with improved path resolution"""
        if path is None:
            return []
        
        project_root = os.path.dirname(os.path.dirname(__file__))
        fixed_files = []
        missing_samples = 0
        max_warn = 10
        
        print(f"[PIADV2Dataset] Reading file list from: {path}")
        
        with open(path, 'r') as f:
            for line_num, raw in enumerate(f, 1):
                line = raw.strip()
                if not line:
                    continue
                    
                original_line = line
                resolved_path = None
                
                # 如果是绝对路径，直接使用
                if os.path.isabs(line):
                    resolved_path = line
                else:
                    # 相对路径解析策略
                    # 1. 首先尝试相对于项目根目录
                    candidate1 = os.path.join(project_root, line)
                    if os.path.exists(candidate1):
                        resolved_path = candidate1
                    else:
                        # 2. 尝试路径修正策略
                        # 处理 Data/Seen/ -> Data/PIADv2/Seen/ 的情况
                        if line.startswith('Data/Seen/'):
                            alt_line = 'Data/PIADv2/Seen/' + line[len('Data/Seen/'):]
                            candidate2 = os.path.join(project_root, alt_line)
                            if os.path.exists(candidate2):
                                resolved_path = candidate2
                        
                        # 处理 Data/Unseen/ -> Data/PIADv2/Unseen/ 的情况
                        elif line.startswith('Data/Unseen/'):
                            alt_line = 'Data/PIADv2/Unseen/' + line[len('Data/Unseen/'):]
                            candidate2 = os.path.join(project_root, alt_line)
                            if os.path.exists(candidate2):
                                resolved_path = candidate2
                        
                        # 处理 Data/Unseen_obj/ -> Data/PIADv2/Unseen_obj/ 的情况
                        elif line.startswith('Data/Unseen_obj/'):
                            alt_line = 'Data/PIADv2/Unseen_obj/' + line[len('Data/Unseen_obj/'):]
                            candidate2 = os.path.join(project_root, alt_line)
                            if os.path.exists(candidate2):
                                resolved_path = candidate2
                        
                        # 如果所有策略都失败，使用原始路径（让错误暴露出来）
                        if resolved_path is None:
                            resolved_path = candidate1
                
                # 转换为绝对路径
                abs_path = os.path.abspath(resolved_path)
                
                # 检查文件是否存在
                if not os.path.exists(abs_path):
                    if missing_samples < max_warn:
                        print(f"[PIADV2Dataset] Warning: File not found at line {line_num}")
                        print(f"  Original: {original_line}")
                        print(f"  Resolved: {abs_path}")
                    missing_samples += 1
                
                fixed_files.append(abs_path)
        
        if missing_samples > 0:
            print(f"[PIADV2Dataset] Total missing files: {missing_samples}/{len(fixed_files)}")
            if missing_samples > max_warn:
                print(f"[PIADV2Dataset] (Only showing first {max_warn} warnings)")
        
        return fixed_files
    
    def _load_point_cloud(self, path):
        """Load point cloud and ground truth mask"""
        try:
            data = np.load(path)
            points = data[:, :3]  # xyz coordinates
            gt_mask = data[:, 3:]  # ground truth mask
            
            # Validate data dimensions
            if points.shape[0] == 0:
                raise ValueError(f"Empty point cloud in {path}")
            if points.shape[1] != 3:
                raise ValueError(f"Invalid point cloud format in {path}: expected 3 coordinates, got {points.shape[1]}")
            if gt_mask.shape[0] != points.shape[0]:
                raise ValueError(f"Point-mask dimension mismatch in {path}: {points.shape[0]} vs {gt_mask.shape[0]}")
            
            # Sample points if necessary
            if len(points) > self.num_points:
                # Use fixed random seed for reproducible sampling during validation
                if self.run_type != 'train':
                    np.random.seed(hash(path) % 2**32)
                indices = np.random.choice(len(points), self.num_points, replace=False)
                points = points[indices]
                gt_mask = gt_mask[indices]
            elif len(points) < self.num_points:
                # Ensure we have at least one point to duplicate
                if len(points) == 0:
                    raise ValueError(f"Cannot sample from empty point cloud in {path}")
                
                # Pad with duplicated points
                diff = self.num_points - len(points)
                # Use modulo to safely handle small point clouds
                indices = np.random.choice(len(points), diff, replace=True)
                points = np.concatenate([points, points[indices]], axis=0)
                gt_mask = np.concatenate([gt_mask, gt_mask[indices]], axis=0)
            
            # Final validation
            assert points.shape[0] == self.num_points, f"Final point count mismatch: {points.shape[0]} != {self.num_points}"
            assert gt_mask.shape[0] == self.num_points, f"Final mask count mismatch: {gt_mask.shape[0]} != {self.num_points}"
            
            return points, gt_mask
            
        except Exception as e:
            print(f"Error loading point cloud from {path}: {e}")
            # Return a fallback point cloud with zeros
            fallback_points = np.zeros((self.num_points, 3), dtype=np.float32)
            fallback_mask = np.zeros((self.num_points, 1), dtype=np.float32)
            return fallback_points, fallback_mask
    
    def _preprocess_points(self, points):
        """Normalize and augment point cloud"""
        # Normalize to unit sphere
        points, _, _ = pc_normalize(points)
        
        # Apply augmentation if training
        if self.use_augmentation and self.run_type == 'train':
            # Add jitter
            points += np.random.normal(0, 0.01, points.shape)
            
            # Random rotation around z-axis
            if np.random.rand() > 0.5:
                angle = np.random.uniform(0, 2 * np.pi)
                cos_angle, sin_angle = np.cos(angle), np.sin(angle)
                rotation_matrix = np.array([
                    [cos_angle, -sin_angle, 0],
                    [sin_angle, cos_angle, 0],
                    [0, 0, 1]
                ])
                points = np.dot(points, rotation_matrix.T)
        
        return points
    
    def _get_image_transform(self):
        """Get image preprocessing transform"""
        if self.run_type == 'train' and self.use_augmentation:
            return transforms.Compose([
                # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def _get_affordance_id(self, img_path):
        """Extract affordance ID from image path"""
        affordance_name = img_path.split('/')[-2]
        return self.affordance_label_list.index(affordance_name)
    
    def _get_instance_id(self, img_path):
        """Extract instance ID from image path"""
        # Extract object name and instance info from path
        parts = img_path.split('/')
        object_name = parts[-4]
        instance_info = parts[-3]
        
        # Create unique instance ID
        instance_key = f"{object_name}_{instance_info}"
        return self.instance_mapping.get(instance_key, 0)
    
    def _create_instance_mapping(self):
        """Create mapping from instance strings to unique IDs"""
        instance_set = set()
        
        # Collect all unique instances
        for img_path in self.img_files:
            parts = img_path.split('/')
            object_name = parts[-4]
            instance_info = parts[-3]
            instance_key = f"{object_name}_{instance_info}"
            instance_set.add(instance_key)
        
        # Create mapping
        return {instance: idx for idx, instance in enumerate(sorted(instance_set))}

    def _create_object_point_map(self):
        """
        Creates a map from an object category to a list of its point cloud indices.
        This is used during training for robust data sampling, ensuring that an image
        is paired with a point cloud of the same object category and affordance.
        Path format is assumed to be .../ObjectClass/Instance/Affordance/xxx.npy
        """
        object_map = {}
        for i, p_path in enumerate(self.point_files):
            try:
                object_name = p_path.split('/')[-4]
                if object_name not in object_map:
                    object_map[object_name] = []
                object_map[object_name].append(i)
            except IndexError:
                # This may happen if a path in the list does not follow the expected format.
                print(f"Warning: Could not parse object name from path: {p_path}. Skipping this entry.")
        return object_map

def get_dataloader(config, split='train'):
    """Create dataloader for specified split"""
    
    # 统一路径处理：优先使用小写文件名，兼容大写文件名
    data_root = config['paths']['data_root']
    
    def get_file_path(base_name, split_suffix):
        """获取文件路径，优先小写，兼容大写"""
        # 优先尝试小写文件名（PIADv2格式）
        lowercase_path = os.path.join(data_root, f'{base_name}_{split_suffix.lower()}.txt')
        if os.path.exists(lowercase_path):
            return lowercase_path
        
        # 尝试大写文件名（PIAD格式）
        uppercase_path = os.path.join(data_root, f'{base_name}_{split_suffix.capitalize()}.txt')
        if os.path.exists(uppercase_path):
            return uppercase_path
        
        # 如果都不存在，返回小写路径（让后续错误处理机制处理）
        return lowercase_path
    
    # Determine paths based on split
    if split == 'train':
        point_path = get_file_path('Point', 'train')
        img_path = get_file_path('Img', 'train')
        use_augmentation = True
    elif split == 'val':
        point_path = get_file_path('Point', 'val')
        img_path = get_file_path('Img', 'val')
        use_augmentation = False
    else:  # test
        point_path = get_file_path('Point', 'test')
        img_path = get_file_path('Img', 'test')
        use_augmentation = False
    
    # Create dataset
    dataset = PIADV2Dataset(
        run_type=split,
        setting_type='Seen',  # Can be configured
        point_path=point_path,
        img_path=img_path,
        image_size=config['data']['image_size'],
        num_points=config['data']['num_points'],
        use_augmentation=use_augmentation
    )
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=(split == 'train'),
        num_workers=4,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader