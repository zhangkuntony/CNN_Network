# CNN Network Project

## Project Overview
This is an image classification project based on Convolutional Neural Networks (CNN), implemented using Python and deep learning frameworks.

## Project Structure
```
CNN_Network/
├── data/          # Dataset directory
├── images/        # Test images and visualization results
└── src/           # Source code directory
    ├── AlexNet.py       # Implements AlexNet architecture with 11 convolutional and fully connected layers
    ├── DataLoader.py    # Loads and preprocesses MNIST dataset
    ├── GoogLeNet.py     # Implements GoogleNet with Inception modules and auxiliary classifiers
    ├── imageAUG.py      # Provides single image and batch image augmentation
    ├── ResNet.py        # Implements residual network with residual blocks and modules
    └── VGG.py           # Implements VGG network using stacked small convolution kernels
```

## Quick Start
1. Clone the repository
```bash
git clone https://github.com/yourusername/CNN_Network.git
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run training
```bash
python src/AlexNet.py
python src/VGG.py
python src/GoogLeNet.py
python src/ResNet.py
```

## Features
- Supports multiple CNN architectures:
  - AlexNet: Classic deep convolutional neural network
  - VGG: Deep network using small convolution kernels
  - GoogLeNet: Network with Inception modules
  - ResNet: Residual network solving deep network training challenges
- Data loading and preprocessing (DataLoader.py)
- Image augmentation (imageAUG.py):
  - Random flipping, cropping
  - Brightness, hue adjustment
  - Batch data augmentation
- Training process visualization
- Model evaluation and testing

## Code Modules
- `AlexNet.py`: Implements AlexNet architecture with 11 convolutional and fully connected layers
- `DataLoader.py`: Loads and preprocesses MNIST dataset
- `GoogLeNet.py`: Implements GoogleNet with Inception modules and auxiliary classifiers
- `imageAUG.py`: Provides single image and batch image augmentation
- `ResNet.py`: Implements residual network with residual blocks and modules
- `VGG.py`: Implements VGG network using stacked small convolution kernels

## Contribution Guidelines
Welcome to submit Pull Requests or report issues

## License
MIT License