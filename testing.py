from diffusion_model import *
from training import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--f", help="directory name", type=str)
parser.add_argument("--v", help="model version", default=final_epoch, type=int)
parser.add_argument("--t", help="type of object to generate", default="Chair", type=str)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 8 # 生成点云数
N = 8192 # 点云点数

name_dict = {"Airplane": 0, "Bag": 1, "Cap": 2, "Car": 3, "Chair": 4, "Earphone": 5, "Guitar": 6, "Knife": 7, "Lamp": 8, "Laptop": 9, "Motorbike": 10, "Mug": 11, "Pistol": 12, "Rocket": 13, "Skateboard": 14, "Table": 15}

save_path = args.f + "_gen/point_cloud"

if __name__ == "__main__":
    state_dict = torch.load(args.f + f"/model_{args.v}.pth", map_location=device)
    model = ModelComposition(latent_dim, max_t, beta_1, beta_T, device)
    model.load_state_dict(state_dict)

    labels = torch.tensor([name_dict[args.t] for _ in range(batch_size)])

    with torch.no_grad():
        print("Sampling...")
        z = torch.randn((batch_size, model.latent_dim)).to(model.device)
        X_0 = model.sample(N, labels, z)
        savePCFromBatch(X_0, save_path=save_path)

    print("Testing ended")