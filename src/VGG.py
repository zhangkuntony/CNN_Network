import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from DataLoader import load_data
import tensorflow as tf

# 构建VGG块
def vgg_block(num_conv, num_filters):
    # 序列模型
    blk = tf.keras.models.Sequential()
    # 遍历卷积层
    for _ in range(num_conv):
        # 设置卷积层
        blk.add(tf.keras.layers.Conv2D(num_filters, kernel_size=3, padding='same', activation='relu'))
    # 池化层
    blk.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    return blk

# 构建模型
def vgg(conv_arch):
    # 序列模型
    net = tf.keras.models.Sequential()
    # 生成卷积部分
    for (num_convs, num_filters) in conv_arch:
        net.add(vgg_block(num_convs, num_filters))

    # 全连接层
    net.add(tf.keras.models.Sequential([
        # 展平
        tf.keras.layers.Flatten(),
        # 全连接层
        tf.keras.layers.Dense(4096, activation='relu'),
        # 随机失活
        tf.keras.layers.Dropout(0.5),
        # 全连接层
        tf.keras.layers.Dense(4096, activation='relu'),
        # 随机失活
        tf.keras.layers.Dropout(0.5),
        # 输出层
        tf.keras.layers.Dense(10, activation='softmax')
    ]))
    return net

def construct_vgg():
    # 设置卷积块的参数
    print('构建VGG网络模型')
    conv_arch = ((2,64),(2,128),(3,256),(3,512),(3,512))
    net = vgg(conv_arch)
    x = tf.random.uniform((1,224,224,1))
    y = net(x)
    print(y)
    print(net.summary())
    return net

def model_complier_fix_evaluate(net, train_images, train_labels, test_images, test_labels):
    # 指定优化器，损失函数和评价指标
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0)
    net.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 模型训练：指定训练数据，batchsize, epoch，验证集
    net.fit(train_images, train_labels, batch_size=128, epochs=3, verbose=1, validation_split=0.1)

    # 模型评估：指定测试数据
    net.evaluate(test_images, test_labels, verbose=1)

def main():
    print("VGG Net")
    net = construct_vgg()
    train_images, train_labels, test_images, test_labels = load_data(256, 128)
    model_complier_fix_evaluate(net, train_images, train_labels, test_images, test_labels)

if __name__ == '__main__':
    main()