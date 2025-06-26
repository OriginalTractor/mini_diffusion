# // TODO 以下是预撰写的项目README.

## 环境配置

Python版本: 3.9.18

模块依赖:
```bash
pip install numpy torch tqdm
```

// TODO: 最后提交前检查是否导入了新模块

## 数据集获取 // TODO

## 训练

运行
```bash
python training.py
```
// TODO: 最后提交前检查是否需要更多参数

## 测试
假定模型保存在`dir_name`目录下. 运行
```bash
python testing.py --f dir_name --v version
```
// TODO: 最后提交前检查是否需要更多参数

以用第`version`个epoch保存的模型进行生成, `version`的默认值为最大训练轮数. 生成的点云以`.obj`文件形式保存于`dir_name + "_gen/point_cloud"`目录下. 

# // TODO 以下是预撰写的实验报告.

# 几何计算前沿 大作业报告

我们的目标是实现基于点云扩散模型的Point-E模型. // TODO: 也有可能是多类生成模型. 

## 分工

// TODO

## 数据读取和预处理

我们的模型数据取自 // TODO: 提供来源

// TODO

## 模型结构

我们的模型结构主要参考论文Diffusion Probabilistic Models for 3D Point Cloud Generation的实现. 

### 子模型

模型中可训练的两个子模型结构如下:

模型$\theta$通过输入隐空间表达$z$, 时间步$t$, 文本信息$\text{emb}$和加噪后点云$X_t$, 输出预测的加噪前点云$X_0$. 模型用输入的隐空间表达和条件表示合成张量`ctx_with_emb` ($z_\text{emb}$):

// TODO: 公式

使$z_\text{emb}$通过融合上下文的线性层:
$$
\text{Gate}(z_\text{emb}) = \text{sigmoid}(W_\text{G}z_\text{emb} + b_\text{G})\\
\text{Bias}(z_\text{emb}) = W_\text{B}z_\text{emb}\\
\text{Linear}'(X, z_\text{emb}) = \text{Gate}(z_\text{emb}) \times \text{Linear}(X) + \text{Bias}(z_\text{emb})
$$

和激活函数$\text{ReLU}$的交替层, 输出对噪声的估计. 实现见于类`PointwiseNet`.

模型$\varphi$预测点云的隐空间表达, 采用不含T-Net结构的PointNet模型. 实现见于类`DistributionEncoder`.

### 训练

我们的扩散模型中设定参数$\beta_t$随时间步$t$线性变化.

在训练中, 给定待拟合的点云$X_0$和文本信息$\text{emb}$, 通过$\varphi$ 预测其在隐空间的表达的分布. 分布用量$z_\mu$, $z_\sigma$表示, 通过

$$z\sim\mathcal{N}(z_\mu, e^{z_\sigma})$$

采样$z$. 

选定随机步长$t\in [0, T)$, 模拟点云前向扩散$t$步, 获得$X_t$. 将$X_t, \beta_t$ (代替$t$), $z$与$\text{emb}$输入$\theta$, 获得噪声的估计$\epsilon_\theta$.

训练目标是优化两项损失:

$$\mathcal{L} = \mathcal{L}_{\text{recons}} + \mathcal{L}_{\text{KL}}$$

其中重建损失$\mathcal{L}_{\text{recons}} = \Vert \epsilon - \epsilon_\theta \Vert^2$, KL散度损失$\mathcal{L}_{\text{KL}}$如下建立:

$$\log p(z) = -\frac{D}{2}\log(2\pi) - \frac{1}{2}|z|^2\\
H(q(z|X_0)) = \frac{D}{2}(1 + \log(2\pi)) + \frac{1}{2}\sum_{j=1}^D \log(\sigma_j^2)\\
\mathcal{L}_{\text{KL}} = -D_{\text{KL}} = \log p(z) + H(q(z|X_0))$$

其中$D$是隐空间维度,$\sigma_j$是标准差$e^{\frac{1}{2}z_\sigma}$的分量. 训练主循环的实现见于`training.py`.

### 测试

测试过程中, 从时间步$T$开始, 基于隐空间随机采样的噪声$z$和给定的文本信息$\text{emb}$, 通过模型预测噪声并逐步去噪. 去噪的算法参考了课件. 

测试主循环的实现见于`testing.py`.

## 实验设置

// TODO: 讲讲参数是怎么设置的

## 结果分析

// TODO