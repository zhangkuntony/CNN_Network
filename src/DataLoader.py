import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def load_data(train_size, test_size):
    # (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # 加载本地数据
    with np.load('../data/mnist.npz') as data:
        train_images = data['x_train']
        train_labels = data['y_train']
        test_images = data['x_test']
        test_labels = data['y_test']

    print("train_images, test_images 在reshape之前的大小：")
    print(train_images.shape)
    print(test_images.shape)

    # 维度调整
    train_images = np.reshape(train_images, (train_images.shape[0], train_images.shape[1], train_images.shape[2], 1))
    test_images = np.reshape(test_images, (test_images.shape[0], test_images.shape[1], test_images.shape[2], 1))

    print("train_images, test_images 在reshape之后的大小：")
    print(train_images.shape)
    print(test_images.shape)

    # 对数据集进行抽样
    train_images, train_labels = get_random_size_data(train_images, train_labels, train_size)
    test_images, test_labels = get_random_size_data(test_images, test_labels, test_size)

    plt.imshow(train_images[87].astype(np.int8).squeeze(), cmap="gray")
    plt.show()
    return train_images, train_labels, test_images, test_labels

def get_random_size_data(images, labels, size):
    # 随机生成index
    index = np.random.randint(0, images.shape[0], size)

    # 选择图像并进行resize
    resized_image = tf.image.resize_with_pad(images[index], 227, 227)
    return resized_image.numpy(), labels[index]
