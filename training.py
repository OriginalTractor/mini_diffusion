from point_cloud_diffusion import *

import time
import argparse
from tqdm import tqdm

import trimesh

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":

    # 超参数
    max_t = 400
    beta_1 = 4e-3
    beta_T = 2e-2

    latent_dim = 256

    learning_rate = 1e-4
    batch_size = 128
    final_epoch = 200

    val_batch_size = 64
    test_freq = 10000
    point_num = 2048 # TODO: (测试集) 点云点数

    timestamp = time.strftime("model_%H_%M_%S_%m_%d_%y", time.localtime())
    save_path = f"models/{timestamp}"
    os.makedirs(save_path, exist_ok=True)

    # Dataloader配置
    data_dir = "data/ModelNet40"
    training_data = PointCloudDataset(data_dir, split="train",use_normals=False)  # 指定训练集
    training_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, collate_fn=torch.utils.data.dataloader.default_collate)

    testing_data = PointCloudDataset(data_dir, split="test",use_normals=False)  # 指定测试集
    testing_dataloader = DataLoader(testing_data, batch_size=val_batch_size, shuffle=False)

    # 模型配置
    model = ModelComposition(latent_dim, max_t, beta_1, beta_T, device).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练
    for epoch in range(1, final_epoch + 1):
        pbar = tqdm(range(len(training_dataloader)))
        tot_loss = 0
        for batch, (X, _) in enumerate(training_dataloader): # 注意：解包为(X, label)
            # X: 输入的点云batch, (B, N, 3)
            pbar.update()

            X = X.to(device)
            loss = model.calLoss(X)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_description(f"Epoch {epoch}/{final_epoch}, Current Loss:{loss.item():>6f}")
        
        pbar.close()
        torch.save(model.state_dict(), save_path + f"/model-{epoch}.pth")
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

    print("Training Ended")