import os
import math
import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import concurrent.futures
from functools import partial
from sklearn.neighbors import KDTree
import random

class PointCloudDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split="train",
        num_points=2048,
        transform=None,
        normalize=True,
        random_rotate=False,
        random_jitter=False,
        random_scale=False,
        cache_size=10000,
        uniform=False,
        normal_channel=True,
        ignore_unknown_classes=True,
        use_normals=False,  # 新增参数，控制是否使用法线
    ):
        """
        初始化ModelNet40格式的点云数据集
        """
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.num_points = num_points
        self.transform = transform
        self.normalize = normalize
        self.random_rotate = random_rotate
        self.random_jitter = random_jitter
        self.random_scale = random_scale
        self.uniform_sample = uniform
        self.normal_channel = normal_channel
        self.ignore_unknown_classes = ignore_unknown_classes
        self.use_normals = use_normals  # 初始化use_normals属性
        
        # 加载类别映射
        self.classes = self._load_classes()
        self.cat2id = {cat: i for i, cat in enumerate(self.classes)}
        self.id2cat = {i: cat for i, cat in enumerate(self.classes)}
        
        # 加载数据文件列表
        self.data_list = self._load_data_list()
        
        # 数据缓存
        self.cache = {}
        self.cache_size = cache_size
        
    def _load_classes(self):
        """加载类别名称列表"""
        classes_file = os.path.join(self.data_dir, 'modelnet40_shape_names.txt')
        with open(classes_file, 'r') as f:
            return [line.strip() for line in f]
    
    def _load_data_list(self):
        """加载数据文件列表，处理可能的类别名称问题"""
        split_file = os.path.join(self.data_dir, f'modelnet40_{self.split}.txt')
        with open(split_file, 'r') as f:
            shape_ids = [line.strip() for line in f]
        
        data_list = []
        for shape_id in shape_ids:
            # 提取类别名称
            parts = shape_id.split('_')
            cat = parts[0]
            
            # 处理特殊类别名称（如 flower_pot, glass_box 等）
            if cat == 'flower' and len(parts) > 1 and parts[1] == 'pot':
                cat = 'flower_pot'
            elif cat == 'glass' and len(parts) > 1 and parts[1] == 'box':
                cat = 'glass_box'
            elif cat == 'night' and len(parts) > 1 and parts[1] == 'stand':
                cat = 'night_stand'
            elif cat == 'range' and len(parts) > 1 and parts[1] == 'hood':
                cat = 'range_hood'
            elif cat == 'tv' and len(parts) > 1 and parts[1] == 'stand':
                cat = 'tv_stand'
            
            # 检查类别是否存在
            if cat not in self.cat2id:
                if self.ignore_unknown_classes:
                    print(f"警告: 忽略未知类别 '{cat}' 中的样本 {shape_id}")
                    continue
                else:
                    raise KeyError(f"未知类别: '{cat}'")
            
            cat_id = self.cat2id[cat]
            file_path = os.path.join(self.data_dir, cat, f'{shape_id}.txt')
            data_list.append((file_path, cat_id))
        
        if len(data_list) == 0:
            raise ValueError(f"在 {split_file} 中没有找到有效类别")
            
        return data_list
    
    def _preload_cache(self, num_workers):
        """预加载部分数据到缓存"""
        cache_size = int(len(self.data_list) * self.cache_rate)
        indices = np.random.choice(len(self.data_list), cache_size, replace=False)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {executor.submit(self._load_data, idx): idx for idx in indices}
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    self.cache[idx] = future.result()
                except Exception as e:
                    print(f"Failed to load data at index {idx}: {e}")
    
    def _load_data(self, idx):
        """加载单个数据样本"""
        file_path, label = self.data_list[idx]
        
        # 读取点云数据
        points = np.loadtxt(file_path, delimiter=',').astype(np.float32)
        
        # 采样点云
        if points.shape[0] > self.num_points:
            if self.uniform_sample:
                points = self._fps_sample(points, self.num_points)
            else:
                indices = np.random.choice(points.shape[0], self.num_points, replace=False)
                points = points[indices]
        else:
            # 不足则重采样
            indices = np.random.choice(points.shape[0], self.num_points, replace=True)
            points = points[indices]
        
        # 提取坐标和法线
        if self.use_normals and points.shape[1] >= 6:
            coords = points[:, :3]
            normals = points[:, 3:6]
            data = np.hstack([coords, normals])
        else:
            data = points[:, :3]
        
        return data, label
    
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
        """获取单个数据样本"""
        if idx in self.cache:
            data, label = self.cache[idx]
        else:
            data, label = self._load_data(idx)
        
        # 转换为PyTorch张量
        data = torch.from_numpy(data).float()
        label = torch.tensor(label, dtype=torch.long)
        
        # 应用变换
        if self.transform:
            data = self.transform(data)
        
        return data, label

def collate_fn(batch, max_points=None):
    """自定义数据批处理函数，支持处理不同点数的点云"""
    if max_points is None:
        return default_collate(batch)
    
    # 随机采样到固定点数
    data_list, label_list = [], []
    for data, label in batch:
        if data.shape[0] > max_points:
            indices = torch.randperm(data.shape[0])[:max_points]
            data = data[indices]
        data_list.append(data)
        label_list.append(label)
    
    # 堆叠成批次
    data_batch = torch.stack(data_list, dim=0)
    label_batch = torch.stack(label_list, dim=0)
    
    return data_batch, label_batch

def get_dataloader(
    root_dir,
    split='train',
    batch_size=32,
    num_points=1024,
    num_workers=4,
    shuffle=True,
    pin_memory=True,
    drop_last=True,
    transform=None,
    use_normals=False,
    cache_rate=0.5,
    uniform_sample=True,
    max_points=None,
):
    """获取数据加载器"""
    dataset = PointCloudDataset(
        root_dir=root_dir,
        split=split,
        num_points=num_points,
        transform=transform,
        use_normals=use_normals,
        cache_rate=cache_rate,
        num_workers=num_workers,
        uniform_sample=uniform_sample,
    )
    
    # 使用自定义collate_fn支持动态采样
    collate = partial(collate_fn, max_points=max_points) if max_points else default_collate
    
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
    
# class LinearWithContext(nn.Module):
#     # 辅助模块. 根据上下文动态调整的线性层, embedding的关键
#     def __init__(self, in_dim, out_dim, ctx_dim):
#         super().__init__()
#         self.layer = nn.Linear(in_dim, out_dim)
#         self.bias_layer = nn.Linear(ctx_dim, out_dim, bias=False)
#         self.gate_layer = nn.Linear(ctx_dim, out_dim)

#     def forward(self, X, ctx):
#         gate = torch.sigmoid(self.gate_layer(ctx)).to(X)
#         bias = self.bias_layer(ctx).to(X)

#         return gate * self.layer(X) + bias

class LinearWithContext(nn.Module):
    def __init__(self, in_dim, out_dim, ctx_dim):
        super().__init__()
        self.layer = nn.Linear(in_dim, out_dim)
        self.bias_layer = nn.Linear(ctx_dim, out_dim, bias=False)
        self.gate_layer = nn.Linear(ctx_dim, out_dim)
    
    def forward(self, X, ctx):
        gate = torch.sigmoid(self.gate_layer(ctx))
        bias = self.bias_layer(ctx)
        
        # 添加维度扩展，使形状变为 (batch_size, 1, out_dim)
        gate = gate.unsqueeze(1)  # (B, 1, D)
        bias = bias.unsqueeze(1)  # (B, 1, D)
        
        # self.layer(X) 形状为 (B, N, D)，现在可以正确广播
        return gate * self.layer(X) + bias

# class PointwiseNet(nn.Module):
#     # 模型\theta, 试图根据隐空间表达去噪
#     def __init__(self, ctx_dim, emb_size = 3):
#         super().__init__()

#         self.net = nn.Sequential(
#             LinearWithContext(3, 128, ctx_dim + emb_size), nn.ReLU(),
#             LinearWithContext(128, 256, ctx_dim + emb_size), nn.ReLU(),
#             LinearWithContext(256, 512, ctx_dim + emb_size), nn.ReLU(),
#             LinearWithContext(512, 256, ctx_dim + emb_size), nn.ReLU(),
#             LinearWithContext(256, 128, ctx_dim + emb_size), nn.ReLU(),
#             LinearWithContext(128, 3, ctx_dim + emb_size)
#         )

#     def forward(self, X, beta, ctx):
#         time_emb = torch.stack([beta, torch.sin(beta), torch.cos(beta)], dim=1).to(X)
#         ctx_with_emb = torch.cat((ctx, time_emb), dim=-1).to(X)

#         return self.net(X, ctx_with_emb)

class ContextWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    
    def forward(self, x, ctx):
        return self.module(x, ctx)

class PointwiseNet(nn.Module):
    def __init__(self, ctx_dim):
        super().__init__()
        self.net = nn.Sequential(
            ContextWrapper(LinearWithContext(3, 128, ctx_dim+3)), nn.ReLU(),
            ContextWrapper(LinearWithContext(128, 256, ctx_dim+3)), nn.ReLU(),
            ContextWrapper(LinearWithContext(256, 512, ctx_dim+3)), nn.ReLU(),
            ContextWrapper(LinearWithContext(512, 256, ctx_dim+3)), nn.ReLU(),
            ContextWrapper(LinearWithContext(256, 128, ctx_dim+3)), nn.ReLU(),
            ContextWrapper(LinearWithContext(128, 3, ctx_dim+3))
        )
    
    def forward(self, X, beta, ctx):
        time_emb = torch.stack([beta, torch.sin(beta), torch.cos(beta)], dim=1)
        ctx_with_emb = torch.cat([ctx, time_emb], dim=1)
        
        # 手动传递上下文
        x = X
        for layer in self.net:
            if isinstance(layer, ContextWrapper):
                x = layer(x, ctx_with_emb)
            else:
                x = layer(x)
        return x

# class Diffusion(nn.Module):
#     # 扩散和计算损失
#     def __init__(self, T, beta_1, beta_T, device):
#         super().__init__()

#         self.device = device
#         self.T = T

#         # betas 线性增加
#         self.betas = torch.linspace(beta_1, beta_T, T).to(device)
#         self.alphas = 1 - self.betas
#         self.alpha_prod = torch.cumprod(self.alphas, dim=0)

#     def diffuseForward(self, X: torch.Tensor, t = None):
#         if t is None:
#             t = torch.randint(1, self.T + 1, (X.shape[0],)).to(self.device)
#         noise = torch.randn_like(X).to(self.device)
#         coeff_t = self.alpha_prod[t]

#         return torch.sqrt(coeff_t) * X + torch.sqrt(1 - coeff_t) * noise, self.betas[t], noise

#     def sample(self, N, ctx: torch.Tensor):
#         raise RuntimeError("Unimplemented")

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
            t = torch.randint(1, self.T+1, (batch_size,)).to(self.device)
        
        # 扩展维度以匹配X的形状
        coeff_t = self.alphas_cumprod[t-1].view(-1, 1, 1)  # 关键修改：添加view(-1, 1, 1)
        noise = torch.randn_like(X).to(self.device)
        X_t = torch.sqrt(coeff_t) * X + torch.sqrt(1 - coeff_t) * noise
        return X_t, self.betas[t-1], noise
    
    def sample(self, N, ctx: torch.Tensor):
        raise RuntimeError("Unimplemented")

class ModelComposition(nn.Module):
    # 组合以上模块
    def __init__(self, latent_dim, T, beta_1, beta_T, device):
        super().__init__()

        self.device = device
        self.latent_dim = latent_dim
        self.T = T
        self.phi = DistributionEncoder(latent_dim)
        self.diffusion = Diffusion(T, beta_1, beta_T, device)
        self.theta = PointwiseNet(latent_dim)

    def calLoss(self, X):

        # 映到隐空间
        z_mu, z_sigma = self.phi(X)
        z = z_mu + torch.exp(0.5 * z_sigma) * torch.randn(z_sigma.size()).to(z_sigma)

        # 加噪
        t = torch.randint(1, self.T + 1, (X.shape[0],)).to(self.device)
        X_t, beta, eps_rand = self.diffusion.diffuseForward(X, t=t)

        r"""以下损失函数为
        $\mathcal{L}_{\text{recons}} = \Vert \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}X_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon, t) \Vert^2$

        $\log p(z) = -\frac{D}{2}\log(2\pi) - \frac{1}{2}|z|^2$

        $H(q(z|X_0)) = \frac{D}{2}(1 + \log(2\pi)) + \frac{1}{2}\sum_{j=1}^D \log(\sigma_j^2)$

        $\mathcal{L}_{\text{KL}} = -D_{\text{KL}} = \log p(z) + H(q(z|X_0))$

        TODO: 为什么KL这样算?
        """

        # 重建损失
        eps_theta = self.theta(X_t, beta, z)
        L_recons = nn.functional.mse_loss(eps_rand, eps_theta)

        # KL损失
        # log_2pi = torch.log(torch.Tensor(2 * torch.pi).to(self.device))
        log_2pi = torch.log(torch.tensor(2 * math.pi, device=self.device))
        log_p_z = -0.5 * (self.latent_dim * log_2pi + (z * z).sum(dim=1))
        H_q = 0.5 * (self.latent_dim * (1 + log_2pi) + z_sigma.sum(dim=1))
        L_kl = -(log_p_z + H_q).mean()

        return L_recons + L_kl
    
    def sample(self, N, ctx: torch.Tensor):
        batch_size = ctx.shape[0]
        X_t = torch.randn([batch_size, N, 3]).to(self.device)
        for t in range(self.T, 0, -1):
            z = torch.randn_like(X_t) if t > 1 else torch.zeros_like(X_t)

            alpha = self.diffusion.alphas[t]
            beta = self.diffusion.betas[t]
            alpha_prod = self.diffusion.alpha_prod[t]
            alpha_prod_next = self.diffusion.alpha_prod[t - 1]
            sigma = torch.sqrt(beta * (1 - alpha_prod_next) / (1 - alpha_prod)) if t > 1 else 0

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
    for b in range(X.shape[0]):
        cloud = X[b]
        cloud_path = save_path + f"_{b}.obj"

        with open(cloud_path, 'w') as f:
            f.write(f"# Vertices: {cloud.shape[0]}\n\n")
            for point in cloud:
                f.write(f"v {point[0]} {point[1]} {point[2]}\n")
