## 环境配置

Python版本: 3.9.18

模块依赖:
```bash
pip install numpy torch tqdm
```

## 数据集获取 

我们的数据集取自 [ShapeNet不带法向量格式](https://gitcode.com/Open-source-documentation-tutorial/a939f/blob/main/README.md). 

确保训练集目录名为`ShapeNet`, 且位于项目中的`data`目录下.

## 训练

运行:
```bash
python training.py [--m model_path]
```
如果传入了`model_path`, 则会读取该路径对应的模型文件, 从该模型开始继续训练.

## 测试
假定模型保存在`dir_name`目录下. 运行:
```bash
python testing.py --f dir_name [--v version] [--t gen_type]
```
以用第`version`个epoch保存的模型进行生成`gen_type`类别的点云. `version`的默认值为最大训练轮数, `gen_type`的默认值为`Chair`. 生成的点云以`.obj`文件形式保存于`dir_name + "_gen/point_cloud"`目录下. 