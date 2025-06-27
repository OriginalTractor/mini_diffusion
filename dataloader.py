import os
import concurrent.futures
from functools import partial

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class PointCloudDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split="train",
        num_points=2048,
        categories={}, # NOTE: 多类生成中它改为了set
        transform=None,
        normalize=True,
        random_rotate=False,
        random_jitter=False,
        random_scale=False,
        cache_size=10000,
        uniform_sample=False,
        use_normals=False,
    ):
        """初始化"""
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.num_points = num_points
        self.categories = categories
        self.transform = transform
        self.normalize = normalize
        self.random_rotate = random_rotate
        self.random_jitter = random_jitter
        self.random_scale = random_scale
        self.uniform_sample = uniform_sample
        self.use_normals = use_normals
        self.cache = {}
        self.cache_size = cache_size
        
        # 加载类别映射
        self.cat2synset, self.synset2cat = self._load_category_mapping()
        self.cat2id = {cat: i for i, cat in enumerate(self.cat2synset.keys())}
        self.id2cat = {i: cat for i, cat in enumerate(self.cat2synset.keys())}
        
        # 加载数据列表
        self.data_list = self._load_data_list()
        
        # 预加载缓存
        self._preload_cache(num_workers=4)
    
    def _load_category_mapping(self):
        """加载synsetoffset到类别的映射"""
        mapping_file = os.path.join(self.data_dir, "synsetoffset2category.txt")
        cat2synset = {}
        synset2cat = {}
        with open(mapping_file, "r") as f:
            for line in f:
                if line.strip():
                    cat, synset = line.strip().split()
                    cat2synset[cat] = synset
                    synset2cat[synset] = cat
        return cat2synset, synset2cat
    
    def _load_data_list(self):
        """加载数据列表, 移除shape_data前缀"""
        for cat in self.categories:
            if cat not in self.cat2synset:
                raise ValueError(f"目标类别 '{cat}' 不在映射文件中")
        
        target_synsets = {self.cat2synset[cat] for cat in self.categories}
        split_file = os.path.join(
            self.data_dir, "train_test_split", f"shuffled_{self.split}_file_list.json"
        )
        
        # 读取JSON文件
        import json
        try:
            with open(split_file, "r") as f:
                shape_ids = json.load(f)
        except Exception as e:
            raise ValueError(f"读取JSON文件失败: {e}")
        
        data_list = []
        
        for shape_id in shape_ids:
            # 移除shape_data/前缀
            if shape_id.startswith("shape_data/"):
                shape_id = shape_id.split("shape_data/")[-1]
            
            # 解析模型ID为synsetoffset/shape_name
            synsetoffset, shape_name = shape_id.split("/", 1)
            
            # 过滤非目标类别
            if synsetoffset not in target_synsets:
                continue
            
            # 构建点云文件路径
            point_file = os.path.join(
                self.data_dir, synsetoffset, "points", f"{shape_name}.pts"
            )
            data_list.append((point_file, self.cat2id[self.synset2cat[synsetoffset]]))
        
        if not data_list:
            raise ValueError(f"未找到有效点云文件")
        
        return data_list
    
    def _preload_cache(self, num_workers):
        """预加载部分数据到缓存"""
        cache_size = min(self.cache_size, len(self.data_list))
        indices = np.random.choice(len(self.data_list), cache_size, replace=False)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {executor.submit(self._load_data, idx): idx for idx in indices}
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                self.cache[idx] = future.result()
    
    def _load_data(self, idx):
        """加载点云文件"""
        file_path, label = self.data_list[idx]

        try:
            # 尝试读取文件
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            points = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # 解析坐标
                parts = line.split()
                if parts and parts[0].upper() == "VRTX":
                    coords = parts[1:4]
                else:
                    coords = parts[:3]

                if len(coords) == 3:
                    try:
                        points.append([float(coords[0]), float(coords[1]), float(coords[2])])
                    except ValueError:
                        print(f"坐标解析失败 - {coords}")

            points = np.array(points, dtype=np.float32)
  
        except Exception as e:
            print(f"加载{file_path}失败: {e}")
            # 生成随机点云作为备用
            points = np.random.uniform(-0.5, 0.5, (self.num_points, 3)).astype(np.float32)

        # 点云采样
        if points.shape[0] > self.num_points:
            if self.uniform_sample:
                points = self._fps_sample(points, self.num_points)
            else:
                indices = np.random.choice(points.shape[0], self.num_points, replace=False)
                points = points[indices]
        else:
            indices = np.random.choice(points.shape[0], self.num_points, replace=True)
            points = points[indices]

        return points, label

    def _fps_sample(self, points, n_samples):
        """最远点采样"""
        centroids = np.zeros(n_samples, dtype=np.int32)
        distances = np.ones(points.shape[0]) * 1e10
        farthest = np.random.randint(0, points.shape[0])

        for i in range(n_samples):
            centroids[i] = farthest
            centroid = points[farthest, :3]
            dist = np.sum((points[:, :3] - centroid) ** 2, axis=1)
            mask = dist < distances
            distances[mask] = dist[mask]
            farthest = np.argmax(distances, axis=0)

        return points[centroids]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if idx in self.cache:
            data, label = self.cache[idx]
        else:
            data, label = self._load_data(idx)

        data = torch.from_numpy(data).float()
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            data = self.transform(data)

        if self.random_rotate:
            data = self._rotate_point_cloud(data)
        if self.random_jitter:
            data = self._jitter_point_cloud(data)
        if self.random_scale:
            data = self._scale_point_cloud(data)

        return data, label

    def _rotate_point_cloud(self, data):
        """随机旋转点云"""
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        rotated_data = np.dot(data.numpy(), rotation_matrix)
        return torch.from_numpy(rotated_data).float()

    def _jitter_point_cloud(self, data):
        """随机抖动点云"""
        jitter = np.random.normal(0, 0.01, size=data.shape)
        jittered_data = data.numpy() + jitter
        jittered_data = np.clip(jittered_data, -0.05, 0.05)
        return torch.from_numpy(jittered_data).float()

    def _scale_point_cloud(self, data):
        """随机缩放点云"""
        scale = np.random.uniform(0.8, 1.2)
        scaled_data = data.numpy() * scale
        return torch.from_numpy(scaled_data).float()

def collate_fn(batch, max_points=None):
    """统一处理不同大小的点云批次"""
    data_list, label_list = [], []
    for data, label in batch:
        if max_points and data.shape[0] > max_points:
            indices = torch.randperm(data.shape[0])[:max_points]
            data = data[indices]
        data_list.append(data)
        label_list.append(label)
    data_batch = torch.stack(data_list, dim=0)
    label_batch = torch.stack(label_list, dim=0)
    return data_batch, label_batch

def get_dataloader(
    data_dir, 
    split='train',
    batch_size=32,
    num_points=1024,
    categories={},
    num_workers=4,
    shuffle=True,
    pin_memory=True,
    drop_last=True,
    transform=None,
    use_normals=False,
    cache_size=10000,
    uniform_sample=True,
):
    """获取指定类别的数据加载器"""
    dataset = PointCloudDataset(
        data_dir=data_dir,
        split=split,
        num_points=num_points,
        categories=categories,
        transform=transform,
        use_normals=use_normals,
        cache_size=cache_size,
        uniform_sample=uniform_sample,
    )

    collate = partial(collate_fn, max_points=num_points)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate,
    )