from diffusion_model import *
from dataloader import *

import time
import argparse
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数
max_t = 600
beta_1 = 4e-3
beta_T = 2e-2

latent_dim = 256

learning_rate = 1e-4
batch_size = 128
final_epoch = 1500

val_batch_size = 64
test_freq = 300
save_freq = 30
point_num = 2048 # 测试时生成的点云点数

categories = {"Airplane", "Car", "Chair", "Table", "Pistol"} # 大作业文档中建议中的一个类是Rifle, 但是我们的文件中没有这个类别, 故用Pistol代替之

if __name__ == "__main__":


    time_stamp = time.strftime("model_%H_%M_%S_%m_%d_%y", time.localtime())
    save_path = f"models/{time_stamp}"
    os.makedirs(save_path, exist_ok=True)

    data_dir = "data/ShapeNet"

    # 训练集
    print("Loading training dataset...")
    training_data = PointCloudDataset(data_dir, split="train", num_points=point_num, categories=categories)
    training_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    # 测试集
    print("Loading testing dataset...")
    testing_data = PointCloudDataset(data_dir, split="test", num_points=point_num, categories=categories)
    testing_dataloader = DataLoader(testing_data, batch_size=val_batch_size, shuffle=False)

    # 模型配置
    model = ModelComposition(latent_dim, max_t, beta_1, beta_T, device).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练
    for epoch in range(1, final_epoch + 1):
        pbar = tqdm(range(len(training_dataloader)))
        tot_loss = 0
        for batch, (X, label) in enumerate(training_dataloader):
            
            pbar.update()

            # TODO: label 处理

            X = X.to(device)
            loss = model.calLoss(X, label)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_description(f"Epoch {epoch}/{final_epoch}, Current Loss:{loss.item():>6f}")
        
        pbar.close()
        if not epoch % save_freq:
            torch.save(model.state_dict(), save_path + f"/model_{epoch}.pth")
        print(f"Epoch {epoch}, Average Loss: {tot_loss / len(training_dataloader):>6f}")

        if not epoch % test_freq: # 测试

            # TODO: 这是无条件生成的测试逻辑
            print(f"Epoch {epoch}, testing")
            for batch, X in enumerate(testing_dataloader):

                # X: (V_B, N, 3), 对每个测试batch计算平均损失
                with torch.no_grad():
                    z = torch.randn((val_batch_size, model.latent_dim)).to(model.device)
                    X_sample = model.sample(point_num, z)
                    CD_list = calCD(X_sample, X)
                    print(f"Batch {batch}, Average CD: {torch.Tensor(CD_list).mean()}")

    print("Training ended")