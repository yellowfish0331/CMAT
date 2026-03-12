"""
PIAD 数据集适配器，用于 LAS 模型训练。
将 PIAD 数据集格式转换为 LAS 模型期望的输入格式。
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import json
import random
import os
import sys

# 添加piad_utils路径
import importlib
_pc_norm_loaded = False
for _candidate in [
    'data.piad_utils.dataset_PIAD',  # 正常包结构
    'piad_utils.dataset_PIAD',       # 相对项目根添加到 PYTHONPATH 后
    'dataset_PIAD'                   # 直接放入 sys.path
]:
    try:
        module = importlib.import_module(_candidate)
        if hasattr(module, 'pc_normalize'):
            pc_normalize = getattr(module, 'pc_normalize')
            _pc_norm_loaded = True
            break
    except Exception:
        continue
if not _pc_norm_loaded:
    raise ImportError('无法导入 pc_normalize, 请检查 piad_utils/dataset_PIAD.py 是否存在')


class PIADDataset(Dataset):
    """
    PIAD数据集适配器类
    
    将 PIAD 数据集的输出格式转换为 LAS 模型期望的输入格式：
    - 'image': (B, 3, H, W) 预处理后的图像张量
    - 'points': (B, N, 3) 点云张量
    - 'gt_mask': (B, N, 1) 真值掩码
    """
    
    def __init__(self, 
                 run_type='train',
                 setting_type='Seen', 
                 point_path=None,
                 img_path=None,
                 box_path=None,
                 image_size=(224, 224),
                 num_points=2048,
                 use_augmentation=True,
                 pair_num=2):
        """
        初始化PIAD数据集
        
        Args:
            run_type: 'train' or 'val'
            setting_type: 'Seen' or 'Unseen'
            point_path: 点云文件路径列表文件
            img_path: 图像文件路径列表文件
            box_path: 边界框文件路径列表文件
            image_size: 图像尺寸 (H, W)
            num_points: 点云采样点数
            use_augmentation: 是否使用数据增强
            pair_num: 训练时每个图像对应的点云数量
        """
        super().__init__()
        
        self.run_type = run_type
        self.setting_type = setting_type
        self.image_size = image_size
        self.num_points = num_points
        self.use_augmentation = use_augmentation
        self.pair_num = pair_num
        
        # PIAD数据集的功能标签列表（17类）
        self.affordance_label_list = [
            'grasp', 'contain', 'lift', 'open', 'lay', 'sit', 'support', 
            'wrapgrasp', 'pour', 'move', 'display', 'push', 'listen', 
            'wear', 'press', 'cut', 'stab'
        ]
        
        # 根据setting_type定义物体类别
        if setting_type == 'Unseen':
            self.object_categories = [
                'Knife', 'Refrigerator', 'Earphone', 'Bag', 'Keyboard', 
                'Chair', 'Hat', 'Door', 'TrashCan', 'Table', 'Faucet', 
                'StorageFurniture', 'Bottle', 'Bowl', 'Display', 'Mug', 'Clock'
            ]
            number_dict = {obj: 0 for obj in self.object_categories}
        else:  # Seen
            self.object_categories = [
                'Earphone', 'Bag', 'Chair', 'Refrigerator', 'Knife', 'Dishwasher', 
                'Keyboard', 'Scissors', 'Table', 'StorageFurniture', 'Bottle', 'Bowl', 
                'Microwave', 'Display', 'TrashCan', 'Hat', 'Clock', 'Door', 'Mug', 
                'Faucet', 'Vase', 'Laptop', 'Bed'
            ]
            number_dict = {obj: 0 for obj in self.object_categories}
        
        # 读取文件路径
        self.img_files = self._read_file_list(img_path)
        self.box_files = self._read_file_list(box_path)
        
        if self.run_type == 'train':
            self.point_files, self.number_dict = self._read_file_list_with_count(point_path, number_dict)
            self.object_train_split = self._create_object_split()
        else:
            self.point_files = self._read_file_list(point_path)
        
        # 图像预处理
        self.image_transform = self._get_image_transform()
        
        print(f"PIAD数据集初始化完成:")
        print(f"  - 运行模式: {run_type}")
        print(f"  - 设置类型: {setting_type}")
        print(f"  - 图像数量: {len(self.img_files)}")
        print(f"  - 点云数量: {len(self.point_files)}")
        print(f"  - 功能类别数: {len(self.affordance_label_list)}")
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index):
        """
        返回字典格式的数据，包含：
        - 'image': 预处理后的图像张量
        - 'points': 点云张量 (N, 3)
        - 'gt_mask': 真值掩码 (N, 1)
        """
        img_path = self.img_files[index]
        box_path = self.box_files[index]
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        
        if self.run_type == 'train':
            # 训练模式：动态采样点云
            # 从图像路径提取物体类别: Data/Seen/Img/Train/Earphone/grasp/Img_Train_Earphone_grasp_1.jpg
            img_parts = img_path.split('/')
            object_name = img_parts[-3]  # Earphone
            
            # 随机选择同类物体的点云
            if object_name in self.object_train_split:
                range_start, range_end = self.object_train_split[object_name]
                point_indices = random.sample(range(range_start, range_end), min(self.pair_num, range_end - range_start))
                
                # 使用第一个点云（可以扩展为多个点云的融合）
                point_path = self.point_files[point_indices[0]]
            else:
                # 如果找不到对应类别，使用索引对应的点云
                point_path = self.point_files[min(index, len(self.point_files) - 1)]
            
            # 处理图像裁剪和边界框
            image, subject_box, object_box = self._get_crop_train(box_path, image)
            image = image.resize(self.image_size)
            image = self.image_transform(image)
            
        else:
            # 验证/测试模式：使用对应索引的点云
            point_path = self.point_files[min(index, len(self.point_files) - 1)]
            
            # 处理边界框（验证模式不裁剪）
            subject_box, object_box = self._get_crop_val(box_path, image)
            image = image.resize(self.image_size)
            image = self.image_transform(image)
        
        # 加载和处理点云
        points, gt_mask = self._load_point_cloud(point_path, img_path)
        points = self._preprocess_points(points)
        
        return {
            'image': image,
            'points': torch.from_numpy(points).float(),
            'gt_mask': torch.from_numpy(gt_mask).float().unsqueeze(-1)  # (N, 1)
        }
    
    def _read_file_list(self, path):
        """读取文件路径列表"""
        if path is None:
            return []
        
        file_list = []
        with open(path, 'r') as f:
            for line in f:
                file_path = line.strip()
                if not file_path:
                    continue
                # 将相对路径转换为绝对路径
                if not os.path.isabs(file_path):
                    project_root = os.path.dirname(os.path.dirname(__file__))
                    if file_path.startswith('Data/'):
                        relative_rest = file_path.split('/', 1)[1]
                        candidate_v2 = os.path.join(project_root, 'Data', 'PIADv2', relative_rest)
                        candidate_v1 = os.path.join(project_root, 'Data', 'PIAD', relative_rest)
                        if os.path.exists(candidate_v2):
                            file_path = candidate_v2
                        else:
                            file_path = candidate_v1
                    else:
                        file_path = os.path.join(project_root, file_path)
                file_list.append(file_path)
        return file_list
    
    def _read_file_list_with_count(self, path, number_dict):
        """读取文件路径列表并统计各类别数量"""
        file_list = []
        with open(path, 'r') as f:
            for line in f:
                file_path = line.strip()
                if not file_path:
                    continue
                if not os.path.isabs(file_path):
                    project_root = os.path.dirname(os.path.dirname(__file__))
                    if file_path.startswith('Data/'):
                        relative_rest = file_path.split('/', 1)[1]
                        candidate_v2 = os.path.join(project_root, 'Data', 'PIADv2', relative_rest)
                        candidate_v1 = os.path.join(project_root, 'Data', 'PIAD', relative_rest)
                        if os.path.exists(candidate_v2):
                            file_path = candidate_v2
                        else:
                            file_path = candidate_v1
                    else:
                        file_path = os.path.join(project_root, file_path)
                # 统计物体类别数量
                path_parts = file_path.split('/')
                if len(path_parts) >= 2:
                    object_name = path_parts[-2]
                    if object_name in number_dict:
                        number_dict[object_name] += 1
                file_list.append(file_path)
        return file_list, number_dict
    
    def _create_object_split(self):
        """创建各物体类别在点云文件列表中的索引范围"""
        object_split = {}
        start_index = 0
        
        for obj_name in self.object_categories:
            if obj_name in self.number_dict:
                end_index = start_index + self.number_dict[obj_name]
                object_split[obj_name] = (start_index, end_index)
                start_index = end_index
        
        return object_split
    
    def _load_point_cloud(self, point_path, img_path):
        """
        加载点云数据和真值掩码
        
        Args:
            point_path: 点云文件路径
            img_path: 图像文件路径（用于提取功能标签）
        
        Returns:
            points: (N, 3) 点云坐标
            gt_mask: (N,) 真值掩码
        """
        # 解析点云文件
        coordinates = []
        with open(point_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    data = line.split(' ')
                    # 前两个是索引，后面是坐标和功能标签
                    coordinate = [float(x) for x in data[2:]]
                    coordinates.append(coordinate)
        
        data_array = np.array(coordinates)
        points = data_array[:, :3]  # xyz坐标
        affordance_labels = data_array[:, 3:]  # 功能标签
        
        # 从图像路径提取功能类别: Data/Seen/Img/Train/Earphone/grasp/Img_Train_Earphone_grasp_1.jpg
        img_parts = img_path.split('/')
        affordance_name = img_parts[-2]  # grasp
        if affordance_name in self.affordance_label_list:
            affordance_index = self.affordance_label_list.index(affordance_name)
            gt_mask = affordance_labels[:, affordance_index]
        else:
            # 如果找不到对应功能，创建全零掩码
            gt_mask = np.zeros(len(points))
        
        # 采样到指定点数
        if len(points) > self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=False)
            points = points[indices]
            gt_mask = gt_mask[indices]
        elif len(points) < self.num_points:
            # 补充点数
            diff = self.num_points - len(points)
            indices = np.random.choice(len(points), diff, replace=True)
            points = np.concatenate([points, points[indices]], axis=0)
            gt_mask = np.concatenate([gt_mask, gt_mask[indices]], axis=0)
        
        return points, gt_mask
    
    def _preprocess_points(self, points):
        """预处理点云：标准化和数据增强"""
        # 标准化到单位球
        points, _, _ = pc_normalize(points)
        
        # 数据增强（仅训练时）
        if self.use_augmentation and self.run_type == 'train':
            # 添加噪声
            points += np.random.normal(0, 0.01, points.shape)
            
            # 绕z轴随机旋转
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
        """获取图像预处理变换"""
        if self.run_type == 'train' and self.use_augmentation:
            return transforms.Compose([
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def _get_crop_train(self, json_path, image):
        """训练模式的图像裁剪处理"""
        try:
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            sub_points, obj_points = [], []
            for box in json_data['shapes']:
                if box['label'] == 'subject':
                    sub_points = box['points']
                elif box['label'] == 'object':
                    obj_points = box['points']
            
            # 如果没有subject框，创建默认框
            if len(sub_points) == 0:
                sub_points = [[0.0, 0.0], [0.0, 0.0]]
            
            # 随机裁剪
            crop_img, crop_subpoints, crop_objpoints = self._random_crop_with_points(
                image, sub_points, obj_points
            )
            
            return crop_img, crop_subpoints, crop_objpoints
            
        except Exception as e:
            print(f"处理边界框文件时出错 {json_path}: {e}")
            # 返回原图像和默认边界框
            return image, [0, 0, 0, 0], [0, 0, 0, 0]
    
    def _get_crop_val(self, json_path, image):
        """验证模式的边界框处理"""
        try:
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            sub_points, obj_points = [], []
            for box in json_data['shapes']:
                if box['label'] == 'subject':
                    sub_points = box['points']
                elif box['label'] == 'object':
                    obj_points = box['points']
            
            # 转换为边界框格式 [x1, y1, x2, y2]
            if len(sub_points) >= 2:
                sub_box = [*sub_points[0], *sub_points[1]]
            else:
                sub_box = [0, 0, 0, 0]
            
            if len(obj_points) >= 2:
                obj_box = [*obj_points[0], *obj_points[1]]
            else:
                obj_box = [0, 0, 0, 0]
            
            return sub_box, obj_box
            
        except Exception as e:
            print(f"处理边界框文件时出错 {json_path}: {e}")
            return [0, 0, 0, 0], [0, 0, 0, 0]
    
    def _random_crop_with_points(self, image, sub_points, obj_points):
        """带点的随机裁剪（参考原PIAD实现）"""
        points = []
        image = np.array(image)
        
        for obj_point in obj_points:
            points.append(obj_point)
        for sub_point in sub_points:
            points.append(sub_point)
        
        h, w = image.shape[0], image.shape[1]
        points = np.array(points, np.int32)
        
        if len(points) == 0:
            # 如果没有点，返回原图像
            return Image.fromarray(image), [0, 0, 0, 0], [0, 0, 0, 0]
        
        min_x, min_y = np.min(points[:, 0]), np.min(points[:, 1])
        max_x, max_y = np.max(points[:, 0]), np.max(points[:, 1])
        
        # 确保边界值有效
        min_x, min_y = max(0, min_x), max(0, min_y)
        max_x, max_y = min(w-1, max_x), min(h-1, max_y)
        
        # 随机裁剪边界
        t = random.randint(0, min_y) if min_y > 0 else 0
        b = random.randint(max_y + 1, h) if max_y + 1 < h else h
        lft = random.randint(0, min_x) if min_x > 0 else 0
        r = random.randint(max_x + 1, w) if max_x + 1 < w else w
        
        # 裁剪图像
        new_img = image[t:b, lft:r, :]
        new_img = Image.fromarray(new_img)
        
        # 调整边界框坐标
        if len(obj_points) >= 2:
            obj_points_array = np.array(obj_points[:2])
            new_objpoints = [[x - lft, y - t] for x, y in obj_points_array]
            obj_LT, obj_RB = new_objpoints[0], new_objpoints[1]
            new_objpoints = [*obj_LT, *obj_RB]
        else:
            new_objpoints = [0, 0, 0, 0]
        
        if len(sub_points) >= 2:
            sub_points_array = np.array(sub_points[:2])
            new_subpoints = [[x - lft, y - t] for x, y in sub_points_array]
            sub_LT, sub_RB = new_subpoints[0], new_subpoints[1]
            new_subpoints = [*sub_LT, *sub_RB]
        else:
            new_subpoints = [0, 0, 0, 0]
        
        return new_img, new_subpoints, new_objpoints


def get_piad_dataloader(config, split='train'):
    """
    创建PIAD数据集的DataLoader
    
    Args:
        config: 配置字典
        split: 'train', 'val', 或 'test'
    
    Returns:
        DataLoader实例
    """
    # 根据split确定文件路径
    data_root = config['paths']['data_root']
    
    if split == 'train':
        point_path = os.path.join(data_root, 'Point_Train.txt')
        img_path = os.path.join(data_root, 'Img_Train.txt') 
        box_path = os.path.join(data_root, 'Box_Train.txt')
        use_augmentation = True
    elif split == 'val':
        point_path = os.path.join(data_root, 'Point_Val.txt')
        img_path = os.path.join(data_root, 'Img_Val.txt')
        box_path = os.path.join(data_root, 'Box_Val.txt')
        use_augmentation = False
    else:  # test
        point_path = os.path.join(data_root, 'Point_Test.txt')
        img_path = os.path.join(data_root, 'Img_Test.txt')
        box_path = os.path.join(data_root, 'Box_Test.txt')
        use_augmentation = False
    
    # 创建数据集
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
    
    # 创建DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=(split == 'train'),
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader
