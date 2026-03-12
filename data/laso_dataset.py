import os
import pandas as pd
import pickle
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    if m < 1e-8:
        return pc, centroid, m
    pc = pc / m
    return pc, centroid, m


class LASODataset(Dataset):
    """
    LASO Dataset for 3D Object Affordance Grounding with Text Prompts
    """

    def __init__(self,
                 run_type='train',
                 data_root=None,
                 num_points=2048,
                 use_augmentation=False,
                 eval_setting='all',  # 'all', 'seen', 'unseen'
                 **kwargs
                 ):
        if data_root is None:
            # Use relative path from project root
            data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data', 'LASO_dataset')
        
        self.run_type = run_type
        self.data_root = data_root
        self.num_points = num_points
        self.use_augmentation = use_augmentation
        self.eval_setting = eval_setting
        
        # Object classes and affordances
        self.classes = ["Bag", "Bed", "Bowl","Clock", "Dishwasher", "Display", "Door", "Earphone", "Faucet",
            "Hat", "StorageFurniture", "Keyboard", "Knife", "Laptop", "Microwave", "Mug",
            "Refrigerator", "Chair", "Scissors", "Table", "TrashCan", "Vase", "Bottle"]
        
        self.affordances = ['lay','sit','support','grasp','lift','contain','open','wrap_grasp','pour', 
                     'move','display','push','pull','listen','wear','press','cut','stab']
        
        # Create mapping dictionaries
        self.cls2idx = {cls.lower(): i for i, cls in enumerate(self.classes)}
        self.aff2idx = {aff: i for i, aff in enumerate(self.affordances)}
        self.idx2cls = {i: cls for cls, i in self.cls2idx.items()}
        self.idx2aff = {i: aff for aff, i in self.aff2idx.items()}

        # Define LASO seen/unseen splits based on original paper
        # Unseen objects and affordances (from LASO evaluation function)
        # Note: Convert to lowercase to match data format
        self.unseen_objects = ['bed', 'dishwasher', 'microwave', 'scissors', 'vase', 'laptop']
        self.unseen_affordances = ['contain', 'lay', 'sit', 'wrap_grasp', 'open', 'display', 'stab', 'grasp', 'press', 'cut']
        
        # Seen objects (all objects except unseen ones)
        all_classes_lower = [cls.lower() for cls in self.classes]
        self.seen_objects = [obj for obj in all_classes_lower if obj not in self.unseen_objects]
        self.seen_affordances = ['grasp', 'contain', 'lift', 'open', 'lay', 'sit', 'support', 'wrap_grasp', 'pour', 
                               'move', 'display', 'push', 'listen', 'wear', 'press', 'cut', 'stab']

        # Load annotations and objects
        with open(os.path.join(data_root, f'anno_{run_type}.pkl'), 'rb') as f:
            all_anno = pickle.load(f)
        
        with open(os.path.join(data_root, f'objects_{run_type}.pkl'), 'rb') as f:
            self.objects = pickle.load(f)
        
        # Filter annotations based on eval_setting
        self.anno = self._filter_annotations(all_anno)

        # Load the CSV file with questions
        self.question_df = pd.read_csv(os.path.join(data_root, 'Affordance-Question.csv'))
        
        print(f"Loaded LASO {run_type} set successfully, length: {len(self.anno)}")
        print(f"Number of unique objects: {len(self.objects)}")
        print(f"Data root: {data_root}")
        
        # Validate data consistency
        self._validate_data()
    
    def _filter_annotations(self, all_anno):
        """Filter annotations based on eval_setting (seen/unseen)"""
        if self.eval_setting == 'all':
            return all_anno
        
        filtered_anno = []
        for item in all_anno:
            obj_class = item['class']
            affordance = item['affordance']
            
            if self.eval_setting == 'seen':
                # Include only seen objects and seen affordances
                if obj_class in self.seen_objects and affordance in self.seen_affordances:
                    filtered_anno.append(item)
            elif self.eval_setting == 'unseen':
                # Include only unseen objects and unseen affordances
                if obj_class in self.unseen_objects and affordance in self.unseen_affordances:
                    filtered_anno.append(item)
        
        print(f"Filtered annotations for {self.eval_setting} setting: {len(filtered_anno)}/{len(all_anno)}")
        return filtered_anno
    
    def _validate_data(self):
        """Validate data consistency"""
        print(f"Validating LASO dataset...")
        
        # Check if all shape_ids in annotations exist in objects
        missing_objects = []
        for item in self.anno:
            shape_id = str(item['shape_id'])
            if shape_id not in self.objects:
                missing_objects.append(shape_id)
        
        if missing_objects:
            print(f"Warning: {len(missing_objects)} shape_ids missing from objects data")
        else:
            print("All shape_ids found in objects data")
        
        # Check class and affordance coverage
        unique_classes = set(item['class'] for item in self.anno)
        unique_affordances = set(item['affordance'] for item in self.anno)
        
        print(f"Unique classes in data: {len(unique_classes)}")
        print(f"Unique affordances in data: {len(unique_affordances)}")
        
        # Check question data
        if not self.question_df.empty:
            print(f"Question data loaded with {len(self.question_df)} entries")
        else:
            print("Warning: Question data is empty")

    def _augment_point_cloud(self, points):
        """Apply data augmentation to point cloud"""
        if not self.use_augmentation:
            return points
        
        # Random rotation around Y-axis
        angle = np.random.uniform(0, 2 * np.pi)
        cos_angle, sin_angle = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([
            [cos_angle, 0, sin_angle],
            [0, 1, 0],
            [-sin_angle, 0, cos_angle]
        ], dtype=np.float32)
        
        points = points @ rotation_matrix.T
        
        # Random jittering
        jitter = np.random.normal(0, 0.01, points.shape).astype(np.float32)
        points = points + jitter
        
        # Random scaling
        scale = np.random.uniform(0.8, 1.2)
        points = points * scale
        
        return points

    def _sample_points(self, points, mask):
        """Sample points to fixed number with robust error handling"""
        n_points = points.shape[0]
        
        # Validate input
        if n_points == 0:
            print(f"Warning: Empty point cloud detected, creating fallback points")
            # Create fallback points
            sampled_points = np.zeros((self.num_points, 3), dtype=np.float32)
            sampled_mask = np.zeros((self.num_points,), dtype=np.float32) if mask is not None else None
            return sampled_points, sampled_mask
        
        try:
            if n_points >= self.num_points:
                # Random sampling without replacement
                if self.run_type != 'train':
                    # Use deterministic sampling for validation/test
                    np.random.seed(42)
                indices = np.random.choice(n_points, self.num_points, replace=False)
            else:
                # Random sampling with replacement to reach target number
                if self.run_type != 'train':
                    np.random.seed(42)
                indices = np.random.choice(n_points, self.num_points, replace=True)
            
            sampled_points = points[indices]
            sampled_mask = mask[indices] if mask is not None else None
            
            # Final validation
            assert sampled_points.shape[0] == self.num_points, f"Point sampling failed: got {sampled_points.shape[0]}, expected {self.num_points}"
            if sampled_mask is not None:
                assert sampled_mask.shape[0] == self.num_points, f"Mask sampling failed: got {sampled_mask.shape[0]}, expected {self.num_points}"
            
            return sampled_points, sampled_mask
            
        except Exception as e:
            print(f"Error in point sampling: {e}")
            # Fallback: create safe default points
            sampled_points = np.zeros((self.num_points, 3), dtype=np.float32)
            sampled_mask = np.zeros((self.num_points,), dtype=np.float32) if mask is not None else None
            return sampled_points, sampled_mask

    def find_question_text(self, object_name, affordance):
        """Find question text for given object and affordance"""
        # Use random question for training, fixed question for testing
        if self.run_type == 'train':
            qid = f'Question{np.random.randint(1, 15)}'
        else:
            qid = 'Question0'
        
        result = self.question_df.loc[
            (self.question_df['Object'] == object_name) & 
            (self.question_df['Affordance'] == affordance), 
            [qid]
        ]
        
        if not result.empty:
            return result.iloc[0][qid]
        else:
            # Fallback to generic question
            return f"Where can I {affordance} this {object_name.lower()}?"
            
    def __getitem__(self, index):
        """Get a single data sample"""
        # Get annotation data
        data = self.anno[index]            
        shape_id = data['shape_id']
        object_class = data['class']
        affordance = data['affordance']
        gt_mask = np.array(data['mask'], dtype=np.float32)
        
        # Get point cloud
        point_cloud = self.objects[str(shape_id)]
        point_cloud = np.array(point_cloud, dtype=np.float32)
        
        # Normalize point cloud
        point_cloud, _, _ = pc_normalize(point_cloud)
        
        # Sample points to fixed number
        point_cloud, gt_mask = self._sample_points(point_cloud, gt_mask)
        
        # Apply augmentation if enabled
        point_cloud = self._augment_point_cloud(point_cloud)
        
        # Get question text
        question_text = self.find_question_text(object_class, affordance)
        
        # Convert to tensors
        points_tensor = torch.from_numpy(point_cloud).float()  # (N, 3)
        mask_tensor = torch.from_numpy(gt_mask).float().unsqueeze(-1)  # (N, 1)
        
        # Create batch dictionary following the training pipeline format
        batch = {
            'points': points_tensor,
            'gt_mask': mask_tensor,
            'text': question_text,  # String for text prompt
            'object_class': object_class,
            'affordance': affordance,
            'class_id': self.cls2idx[object_class.lower()],
            'affordance_id': self.aff2idx[affordance],
            'shape_id': shape_id
        }
        
        return batch

    def __len__(self):
        return len(self.anno)


def collate_fn(batch):
    """
    Custom collate function for LASO dataset
    Handles batching of point clouds and text prompts
    """
    # Stack tensors
    points = torch.stack([item['points'] for item in batch])  # (B, N, 3)
    gt_masks = torch.stack([item['gt_mask'] for item in batch])  # (B, N, 1)
    
    # Collect text prompts as list
    texts = [item['text'] for item in batch]
    
    # Collect other metadata
    object_classes = [item['object_class'] for item in batch]
    affordances = [item['affordance'] for item in batch]
    class_ids = torch.tensor([item['class_id'] for item in batch])
    affordance_ids = torch.tensor([item['affordance_id'] for item in batch])
    shape_ids = [item['shape_id'] for item in batch]
    
    batch_dict = {
        'points': points,
        'gt_mask': gt_masks,
        'text': texts,
        'object_class': object_classes,
        'affordance': affordances,
        'class_id': class_ids,
        'affordance_id': affordance_ids,
        'shape_id': shape_ids
    }
    
    return batch_dict


def get_laso_dataloader(config, split='train', eval_setting='all'):
    """
    Create LASO dataloader for training/testing
    
    Args:
        config: Configuration dictionary
        split: 'train', 'val', or 'test'
        eval_setting: 'all', 'seen', 'unseen' (only applies to test/val splits)
    
    Returns:
        DataLoader instance
    """
    # Map split names
    if split == 'val':
        split = 'val'  # LASO has val split
    elif split == 'test':
        split = 'test'
    
    # For training, always use 'all' setting; for eval, use specified setting
    final_eval_setting = 'all' if split == 'train' else eval_setting
    
    # Create dataset
    dataset = LASODataset(
        run_type=split,
        data_root=config['paths'].get('laso_data_root', None),
        num_points=config['data']['num_points'],
        use_augmentation=(split == 'train'),
        eval_setting=final_eval_setting
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=(split == 'train'),
        num_workers=config['training'].get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader


if __name__ == '__main__':
    # Test the dataset
    dataset = LASODataset('train')
    print(f"Dataset length: {len(dataset)}")
    
    # Test a sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Points shape: {sample['points'].shape}")
    print(f"Mask shape: {sample['gt_mask'].shape}")
    print(f"Text prompt: {sample['text']}")
    print(f"Object class: {sample['object_class']}")
    print(f"Affordance: {sample['affordance']}")
    
    # Test dataloader
    config = {
        'data': {'num_points': 2048},
        'training': {'batch_size': 4, 'num_workers': 0},
        'paths': {}
    }
    
    dataloader = get_laso_dataloader(config, 'train')
    batch = next(iter(dataloader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Batch points shape: {batch['points'].shape}")
    print(f"Batch masks shape: {batch['gt_mask'].shape}")
    print(f"Batch texts: {batch['text'][:2]}")  # Print first 2 texts