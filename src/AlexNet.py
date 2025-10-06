import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from DataLoader import load_data
import tensorflow as tf

def alex_net_model():
    # 构建模型
    net = tf.keras.models.Sequential([
        # 构建卷积核大小为11*11，个数为96个，步长为4的卷积层，使用ReLU激活函数
        tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4, activation="relu"),

        # 构建3*3大小，步长为2的池化层
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),

        # 构建卷积核大小为5*5，个数为256个，padding为same（填充为2），步长为1的卷积层，使用ReLU激活函数
        tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding="same", activation="relu"),

        # 构建3*3大小，步长为2的池化层
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),

        # 卷积：384 3*3 1 RELU same
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding="same", activation="relu"),
        # 卷积：384 3*3 1 RELU same
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding="same", activation="relu"),
        # 卷积：256 3*3 1 RELU same
        tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", activation="relu"),

        # 池化：3*3 2
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),

        # 展开
        tf.keras.layers.Flatten(),

        # 全连接层: 4096 ReLU
        tf.keras.layers.Dense(units=4096, activation="relu"),
        # 随机失活
        tf.keras.layers.Dropout(rate=0.5),

        # 全连接层: 4096 ReLU
        tf.keras.layers.Dense(units=4096, activation="relu"),
        # 随机失活
        tf.keras.layers.Dropout(rate=0.5),

        # 输出层: 对手写数字0-9分类，共10个类别
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    x = tf.random.uniform((1, 227, 227, 1))
    y = net(x)
    print(net.summary())
    print(y)
    return net

def model_compile_fit(net, train_images, train_labels, test_images, test_labels):
    net.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=["accuracy"])
    net.fit(train_images, train_labels, batch_size=64, epochs=5, validation_split=0.2, verbose=1)
    net.evaluate(test_images, test_labels, verbose=1)

def main():
    net = alex_net_model()
    train_images, train_label, test_images, test_label = load_data(1024, 128)
    model_compile_fit(net, train_images, train_label, test_images, test_label)

if __name__ == '__main__':
    main()