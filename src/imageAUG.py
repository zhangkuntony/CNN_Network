import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def image_augmentation():
    print('tf.image进行增强')
    cat = plt.imread('../images/cat.jpg')
    print('显示原图')
    plt.imshow(cat)
    plt.show()

    print('左右翻转')
    cat1 = tf.image.random_flip_left_right(cat)
    plt.imshow(cat1)
    plt.show()

    print('上下翻转')
    cat2 = tf.image.random_flip_up_down(cat)
    plt.imshow(cat2)
    plt.show()

    print('图像裁剪')
    cat3 = tf.image.random_crop(cat, (200, 200, 3))
    plt.imshow(cat3)
    plt.show()

    print('调整亮度')
    cat4 = tf.image.random_brightness(cat, 0.2)
    plt.imshow(cat4)
    plt.show()

    print('调整色调')
    cat5 = tf.image.random_hue(cat, 0.2)
    plt.imshow(cat5)
    plt.show()

def imagedata_generator():
    with np.load('../data/mnist.npz') as data:
        x_train = data['x_train']
        y_train = data['y_train']
        x_test = data['x_test']
        y_test = data['y_test']

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=3,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.3,0.5],
        shear_range=1,
        zoom_range=0.1,
        vertical_flip=True,
        rescale=1.0 / 255.0
    )

    for x,y in datagen.flow(x_train, y_train, batch_size=9):
        plt.figure(figsize=(8,8))
        for i in range(0,9):
            # plt.subplot(300+1+i)
            plt.subplot(3,3,i+1)
            plt.imshow(x[i].reshape(28,28),cmap='gray')
            plt.title(y[i])
        plt.show()
        break

def main():
    image_augmentation()
    imagedata_generator()

if __name__ == '__main__':
    main()