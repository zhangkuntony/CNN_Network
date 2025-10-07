# CNN网络项目

## 项目简介
这是一个基于卷积神经网络(CNN)的图像分类项目，使用Python和深度学习框架实现。

## 项目结构
```
CNN_Network/
├── data/          # 数据集存放目录
├── images/        # 测试图片和结果可视化
└── src/           # 源代码目录
    ├── AlexNet.py       # 实现AlexNet网络结构，包含11层卷积和全连接层
    ├── DataLoader.py    # 加载MNIST数据集并进行预处理
    ├── GoogLeNet.py     # 实现GoogleNet网络，包含Inception模块和辅助分类器
    ├── imageAUG.py      # 提供单张图片和批量图片的数据增强功能
    ├── ResNet.py        # 实现残差网络，包含残差块和残差模块
    └── VGG.py           # 实现VGG网络，使用多个小卷积核堆叠
```

## 快速开始
1. 克隆仓库
```bash
git clone https://github.com/yourusername/CNN_Network.git
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 运行程序：
```bash
python src/AlexNet.py
python src/VGG.py
python src/GoogLeNet.py
python src/ResNet.py
```

## 功能特性
- 支持多种CNN架构:
  - AlexNet: 经典的深度卷积神经网络
  - VGG: 使用小卷积核的深度网络
  - GoogLeNet: 包含Inception模块的网络
  - ResNet: 残差网络，解决深度网络训练难题
- 数据加载和预处理(DataLoader.py)
- 图像增强功能(imageAUG.py):
  - 随机翻转、裁剪
  - 亮度、色调调整
  - 批量数据增强
- 可视化训练过程
- 模型评估和测试

## 代码模块说明
- `AlexNet.py`: 实现AlexNet网络结构，包含11层卷积和全连接层
- `DataLoader.py`: 加载MNIST数据集并进行预处理
- `GoogLeNet.py`: 实现GoogleNet网络，包含Inception模块和辅助分类器
- `imageAUG.py`: 提供单张图片和批量图片的数据增强功能
- `ResNet.py`: 实现残差网络，包含残差块和残差模块
- `VGG.py`: 实现VGG网络，使用多个小卷积核堆叠
