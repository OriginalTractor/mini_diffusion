import os
import math
import torch
from torch import nn

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

class PointwiseNet(nn.Module):
    # 模型\theta, 试图根据隐空间表达去噪
    def __init__(self, ctx_dim):
        super().__init__()
        self.emb_dim = 11
        self.net = nn.Sequential(
            LinearWithContext(3, 128, ctx_dim + self.emb_dim), nn.ReLU(),
            LinearWithContext(128, 256, ctx_dim + self.emb_dim), nn.ReLU(),
            LinearWithContext(256, 512, ctx_dim + self.emb_dim), nn.ReLU(),
            LinearWithContext(512, 256, ctx_dim + self.emb_dim), nn.ReLU(),
            LinearWithContext(256, 128, ctx_dim + self.emb_dim), nn.ReLU(),
            LinearWithContext(128, 3, ctx_dim + self.emb_dim)
        )
        self.label_emb = nn.Embedding(16, 8)

    def forward(self, X, beta, labels: torch.Tensor, ctx: torch.Tensor):
        # X: (B, N, 3), 
        # beta: () [batch内时间步相同] or (B,) [batch内时间步不同] 
        # labels: (B,)
        # ctx: (B, latent_D)

        B, _ = ctx.shape

        # (B,)
        if beta.dim() == 0:
            beta = beta.view((1,)).repeat(B)

        # (B, 3), (B, 8)
        label_emb = self.label_emb(labels).to(X)
        time_emb = torch.stack([beta, torch.sin(beta), torch.cos(beta)], dim=-1).to(X)

        # (B, latent_D + 11)
        ctx_with_emb = torch.cat([ctx, time_emb, label_emb], dim=1)
        
        for layer in self.net:
            X = layer(X, ctx_with_emb) if isinstance(layer, LinearWithContext) else layer(X)

        return X

class Diffusion(nn.Module):
    # 扩散模块
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

    def calLoss(self, X, labels: torch.Tensor):

        X = X.to(self.device)

        # 映到隐空间
        z_mu, z_sigma = self.phi(X)
        z = z_mu + torch.exp(0.5 * z_sigma) * torch.randn(z_sigma.size()).to(z_sigma)

        # 加噪
        t = torch.randint(0, self.T, (X.shape[0],)).to(self.device)
        X_t, beta, eps_rand = self.diffusion.diffuseForward(X, t=t)

        # 重建损失
        eps_theta = self.theta(X_t, beta, labels, z)
        L_recons = nn.functional.mse_loss(eps_rand, eps_theta)

        # KL损失
        log_2pi = torch.log(torch.tensor(2 * math.pi, device=self.device))
        log_p_z = -0.5 * (self.latent_dim * log_2pi + (z * z).sum(dim=1))
        H_q = 0.5 * (self.latent_dim * (1 + log_2pi) + z_sigma.sum(dim=1))
        L_kl = -(log_p_z + H_q).mean()

        return L_recons + L_kl
    
    def sample(self, N, labels: torch.Tensor, ctx: torch.Tensor):
        batch_size = ctx.shape[0]
        X_t = torch.randn([batch_size, N, 3]).to(self.device)
        for t in range(self.T - 1, -1, -1):
            z = torch.randn_like(X_t) if t > 0 else torch.zeros_like(X_t)

            alpha = self.diffusion.alphas[t]
            beta = self.diffusion.betas[t]
            alpha_prod = self.diffusion.alphas_cumprod[t]
            alpha_prod_next = self.diffusion.alphas_cumprod[t - 1] if t > 0 else 1

            sigma = torch.sqrt(beta * (1 - alpha_prod_next) / (1 - alpha_prod))

            eps_theta = self.theta(X_t, beta, labels, ctx)
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