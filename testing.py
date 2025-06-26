from point_cloud_diffusion import *
from training import *

import time
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--f", help="dir name", default="", type=str)
parser.add_argument("--v", help="version", default=final_epoch, type=int)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 8 # 生成点云数
N = 2048 # 点云点数

save_path = args.f + "_gen/point_cloud"

if __name__ == "__main__":
    state_dict = torch.load(args.f + f"/model_{args.v}.pth", map_location=device)
    model = ModelComposition(latent_dim, max_t, beta_1, beta_T, device)
    model.load_state_dict(state_dict)

    # TODO: 这是无条件生成的逻辑
    with torch.no_grad():
        print("Sampling...")
        z = torch.randn((batch_size, model.latent_dim)).to(model.device)
        X_0 = model.sample(N, z)
        savePCFromBatch(X_0, save_path=save_path)

    print("Testing ended")