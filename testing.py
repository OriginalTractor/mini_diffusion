from point_cloud_diffusion import *

import time
import argparse
from tqdm import tqdm

import trimesh

parser = argparse.ArgumentParser()
parser.add_argument("--f", help="file name", default="", type=str)
args = parser.parse_args()

batch_size = 8 # 生成点云数
N = 2048 # 点云点数

save_path = args.f + "_gen/point_cloud"

if __name__ == "__main__":
    model: ModelComposition = torch.load(args.f)

    # TODO: 这是无条件生成的逻辑
    with torch.no_grad():
        z = torch.randn((batch_size, model.latent_dim)).to(model.device)
        X_0 = model.sample(N, z)
        savePCFromBatch(X_0, save_path=save_path)