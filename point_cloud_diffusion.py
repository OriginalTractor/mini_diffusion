import os
import math
import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import concurrent.futures
from functools import partial

class PointCloudDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split="train",
        num_points=2048,
        category=None,
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
        self.category = category
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
        """加载数据列表（强制移除shape_data前缀）"""
        if self.category not in self.cat2synset:
            raise ValueError(f"目标类别 '{self.category}' 不在映射文件中")
        
        target_synset = self.cat2synset[self.category]
        split_file = os.path.join(
            self.data_dir, "train_test_split", f"shuffled_{self.split}_file_list.json"
        )
        
        # 读取JSON文件
        import json
        try:
            with open(split_file, "r") as f:
                shape_ids = json.load(f)
            #print(f"从{split_file}读取到{len(shape_ids)}个模型ID")
        except Exception as e:
            raise ValueError(f"读取JSON文件失败：{e}")
        
        data_list = []
        valid_count = 0
        missing_count = 0
        non_chair_count = 0
        prefix_removed_count = 0
        
        for shape_id in shape_ids:
            # 强制移除shape_data/前缀
            if shape_id.startswith("shape_data/"):
                shape_id = shape_id.split("shape_data/")[-1]
                prefix_removed_count += 1
                #print(f"已移除shape_data前缀: {shape_id}")
            
            # 解析模型ID为synsetoffset/shape_name
            try:
                synsetoffset, shape_name = shape_id.split("/", 1)  # 最多分割一次
            except ValueError:
                print(f"警告：无效模型ID格式 '{shape_id}'，应为'synsetoffset/shape_name'，跳过")
                continue
            
            # 过滤非目标类别
            if synsetoffset != target_synset:
                non_chair_count += 1
                continue
            
            # 构建点云文件路径
            point_file = os.path.join(
                self.data_dir, synsetoffset, "points", f"{shape_name}.pts"
            )
            
            # 检查文件存在性
            if os.path.exists(point_file):
                data_list.append((point_file, self.cat2id[self.category]))
                valid_count += 1
            else:
                missing_count += 1
                print(f"警告：点云文件缺失 - {point_file}")
        
        if not data_list:
            raise ValueError(f"未找到有效点云文件，请检查路径和数据完整性")
        
        return data_list
    
    def _preload_cache(self, num_workers):
        """预加载部分数据到缓存"""
        cache_size = min(self.cache_size, len(self.data_list))
        indices = np.random.choice(len(self.data_list), cache_size, replace=False)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {executor.submit(self._load_data, idx): idx for idx in indices}
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    self.cache[idx] = future.result()
                except Exception as e:
                    print(f"加载索引 {idx} 数据失败: {e}")
    
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
                        print(f"警告：坐标解析失败 - {coords}")
            
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
        """最远点采样(FPS)"""
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
    category=None,
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
        category=category,
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
    
class DistributionEncoder(nn.Module):
    # 编码器 (模型\phi), 采用不含T-Net的PointNet结构

    def __init__(self, out_dim):
        super().__init__()

        self.out_dim = out_dim

        # (B, 3, N) -> (B, 512, N)
        self.net0 = nn.Sequential( # Conv + BN便于归一化点云
            nn.Conv1d(3, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 512, 1), nn.BatchNorm1d(512)
        )

        # (B, 512) -> (B, out_dim)
        self.net_mu = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256), 
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), 
            nn.Linear(128, out_dim)
        )

        # (B, 512) -> (B, out_dim)
        self.net_sigma = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256), 
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), 
            nn.Linear(128, out_dim)
        )
    
    def forward(self, X: torch.Tensor):
        # (B, N, 3) -> tuple((B, out_dim), (B, out_dim))
        X = X.transpose(1, 2)
        X = self.net0(X)
        X = torch.max(X, 2).values

        mu = self.net_mu(X).to(X)
        sigma = self.net_sigma(X).to(X)

        return mu, sigma
    
class LinearWithContext(nn.Module):
    def __init__(self, in_dim, out_dim, ctx_dim):
        super().__init__()
        self.layer = nn.Linear(in_dim, out_dim)
        self.bias_layer = nn.Linear(ctx_dim, out_dim, bias=False)
        self.gate_layer = nn.Linear(ctx_dim, out_dim)
    
    def forward(self, X, ctx):
        # X: (B, N, D)

        # (B, 1, D)
        gate = torch.sigmoid(self.gate_layer(ctx)).unsqueeze(1)
        bias = self.bias_layer(ctx).unsqueeze(1)
        
        return gate * self.layer(X) + bias

# TODO: 不应该有这个设计, 应当在point-e版本弃用它
class ContextWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    
    def forward(self, x, ctx):
        return self.module(x, ctx)

class PointwiseNet(nn.Module):
    # 模型\theta, 试图根据隐空间表达去噪
    def __init__(self, ctx_dim):
        super().__init__()
        self.net = nn.Sequential(# TODO: 见上个TODO
            ContextWrapper(LinearWithContext(3, 128, ctx_dim+3)), nn.ReLU(),
            ContextWrapper(LinearWithContext(128, 256, ctx_dim+3)), nn.ReLU(),
            ContextWrapper(LinearWithContext(256, 512, ctx_dim+3)), nn.ReLU(),
            ContextWrapper(LinearWithContext(512, 256, ctx_dim+3)), nn.ReLU(),
            ContextWrapper(LinearWithContext(256, 128, ctx_dim+3)), nn.ReLU(),
            ContextWrapper(LinearWithContext(128, 3, ctx_dim+3))
        )
    
    def forward(self, X, beta, ctx: torch.Tensor):
        # X: (B, N, 3), beta: (), ctx: (B, latent_D)

        B, _ = ctx.shape
        beta = beta.view((1,))

        # (B, 3)
        time_emb = torch.stack([beta, torch.sin(beta), torch.cos(beta)], dim=-1).repeat(B, 1)

        # (B, latent_D + 3)
        ctx_with_emb = torch.cat([ctx, time_emb], dim=1)
        
        for layer in self.net:
            X = layer(X, ctx_with_emb) if isinstance(layer, ContextWrapper) else layer(X)# TODO: 见上个TODO

        return X

class Diffusion(nn.Module):
    def __init__(self, T, beta_1, beta_T, device):
        super().__init__()
        self.T = T
        self.device = device
        self.betas = torch.linspace(beta_1, beta_T, T).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    
    def diffuseForward(self, X, t=None):
        batch_size = X.shape[0]
        if t is None:
            t = torch.randint(0, self.T, (batch_size,)).to(self.device)
        
        coeff_t = self.alphas_cumprod[t].view(-1, 1, 1)
        noise = torch.randn_like(X).to(self.device)
        X_t = torch.sqrt(coeff_t) * X + torch.sqrt(1 - coeff_t) * noise
        return X_t, self.betas[t], noise

class ModelComposition(nn.Module):
    # 组合以上模块
    def __init__(self, latent_dim, T, beta_1, beta_T, device):
        super().__init__()

        self.device = device
        self.latent_dim = latent_dim
        self.T = T
        self.phi = DistributionEncoder(latent_dim).to(device)
        self.diffusion = Diffusion(T, beta_1, beta_T, device)
        self.theta = PointwiseNet(latent_dim).to(device)

    def calLoss(self, X):

        X = X.to(self.device)

        # 映到隐空间
        z_mu, z_sigma = self.phi(X)
        z = z_mu + torch.exp(0.5 * z_sigma) * torch.randn(z_sigma.size()).to(z_sigma)

        # 加噪
        t = torch.randint(0, self.T, (X.shape[0],)).to(self.device)
        X_t, beta, eps_rand = self.diffusion.diffuseForward(X, t=t)

        # 重建损失
        eps_theta = self.theta(X_t, beta, z)
        L_recons = nn.functional.mse_loss(eps_rand, eps_theta)

        # KL损失
        log_2pi = torch.log(torch.tensor(2 * math.pi, device=self.device))
        log_p_z = -0.5 * (self.latent_dim * log_2pi + (z * z).sum(dim=1))
        H_q = 0.5 * (self.latent_dim * (1 + log_2pi) + z_sigma.sum(dim=1))
        L_kl = -(log_p_z + H_q).mean()

        return L_recons + L_kl
    
    def sample(self, N, ctx: torch.Tensor):
        batch_size = ctx.shape[0]
        X_t = torch.randn([batch_size, N, 3]).to(self.device)
        for t in range(self.T - 1, -1, -1):
            z = torch.randn_like(X_t) if t > 0 else torch.zeros_like(X_t)

            alpha = self.diffusion.alphas[t]
            beta = self.diffusion.betas[t]
            alpha_prod = self.diffusion.alphas_cumprod[t]
            alpha_prod_next = self.diffusion.alphas_cumprod[t - 1] if t > 0 else 1
            sigma = torch.sqrt(beta * (1 - alpha_prod_next) / (1 - alpha_prod))

            eps_theta = self.theta(X_t, beta, ctx)
            X_t = (X_t - beta / torch.sqrt(1 - alpha_prod) * eps_theta) / torch.sqrt(alpha) + sigma * z

        return X_t

def calCD(sample_coluds: torch.Tensor, ref_clouds: torch.Tensor):
    # 倒角距离
    batch = sample_coluds.shape[0]
    CDs = []
    for b in range(batch):
        sample, ref = sample_coluds[b], ref_clouds[b]
        dist_mat = torch.cdist(sample, ref)
        CDs.append(torch.mean(torch.min(dist_mat, dim=1).values ** 2) + 
                   torch.mean(torch.min(dist_mat, dim=0).values ** 2))
        
    return CDs

def savePCFromBatch(X: torch.Tensor, save_path):
    # X: (B, N, 3)
    print("Saving...")
    for b in range(X.shape[0]):
        cloud = X[b]
        cloud_path = save_path + f"_{b}.obj"

        os.makedirs(save_path, exist_ok=True)
        with open(cloud_path, 'w') as f:
            f.write(f"# Vertices: {cloud.shape[0]}\n\n")
            for point in cloud:
                f.write(f"v {point[0]} {point[1]} {point[2]}\n")
