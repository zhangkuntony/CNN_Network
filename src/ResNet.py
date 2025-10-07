import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from DataLoader import load_data
from tensorflow.keras import layers, activations
import tensorflow as tf

# 残差块
class Residual(tf.keras.Model):
    # 定义网络结构
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super(Residual, self).__init__()
        # 卷积层
        self.conv1 = layers.Conv2D(num_channels, padding='same', kernel_size=3, strides=strides)
        # 卷积层
        self.conv2 = layers.Conv2D(num_channels, kernel_size=3, padding='same')
        # 是否使用1*1的卷积
        if use_1x1conv:
            self.conv3 = layers.Conv2D(num_channels, kernel_size=1, strides=strides)
        else:
            self.conv3 = None
        # BN层
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()

    # 定义前向传播过程
    def call(self, x):
        y = activations.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        outputs = activations.relu(y+x)
        return outputs

# 残差模块
class ResnetBlock(tf.keras.layers.Layer):
    # 定义所需的网络结构
    def __init__(self, num_channels, num_res, first_block=False):
        super(ResnetBlock, self).__init__()
        # 存储残差块
        self.listLayers = []
        # 遍历残差数目生成模块
        for i in range(num_res):
            # 如果是第一个残差块而且不是第一个模块时
            if i == 0 and not first_block:
                self.listLayers.append(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                self.listLayers.append(Residual(num_channels))

    # 定义前向传播
    def call(self, x):
        for layer in self.listLayers:
            x = layer(x)
        return x

    # 添加显式的__call__方法，解决告警：'Inception' object is not callable
    def __call__(self, inputs, *args, **kwargs):
        return super().__call__(inputs, *args, **kwargs)

# 构建resNet网络
class ResNet(tf.keras.Model):
    # 定义网络的构成
    def __init__(self, num_blocks):
        super(ResNet, self).__init__()
        # 输入层
        self.conv = layers.Conv2D(64, kernel_size=7, strides=2, padding='same')
        # BN层
        self.bn = layers.BatchNormalization()
        # 激活层
        self.relu = layers.Activation('relu')
        # 池化
        self.mp = layers.MaxPool2D(pool_size=3, strides=2, padding='same')
        # 残差模块
        self.res_block1 = ResnetBlock(64, num_blocks[0], first_block=True)
        self.res_block2 = ResnetBlock(128, num_blocks[1])
        self.res_block3 = ResnetBlock(256, num_blocks[2])
        self.res_block4 = ResnetBlock(512, num_blocks[3])
        # GAP
        self.gap = layers.GlobalAveragePooling2D()
        # 全连接层
        self.fc = layers.Dense(
            units=10,
            activation=tf.keras.activations.softmax
        )

    # 定义前向传播过程
    def call(self, x):
        # 输入部分的传输过程
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.mp(x)
        # block
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        # 输出部分的传输
        x = self.gap(x)
        x = self.fc(x)
        return x

    # 添加显式的__call__方法，解决告警：'Inception' object is not callable
    def __call__(self, inputs, *args, **kwargs):
        return super().__call__(inputs, *args, **kwargs)


def res_net_creator():
    res_net = ResNet([2,2,2,2])
    x = tf.random.uniform((1,224,224,1))
    y = res_net(x)
    print(y)
    print(res_net.summary())
    return res_net

def model_compile_fit_evaluate(model, train_images, train_labels, test_images, test_labels):
    # 模型编译：指定优化器，损失函数和评价指标
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 模型训练：指定训练数据, batchsize, epoch, 验证集
    model.fit(train_images, train_labels, batch_size=128, epochs=3, verbose=1, validation_split=0.1)

    # 模型评估：指定测试数据
    evaluate = model.evaluate(test_images, test_labels, verbose=1)
    print(evaluate)

def main():
    print("ResNet")
    net = res_net_creator()
    train_images, train_labels, test_images, test_labels = load_data(256,128)
    model_compile_fit_evaluate(net, train_images, train_labels, test_images, test_labels)

if __name__ == '__main__':
    main()
