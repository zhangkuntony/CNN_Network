import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from DataLoader import load_data
import tensorflow as tf

class Inception(tf.keras.layers.Layer):
    # 设置模块的构成
    def __init__(self,c1,c2,c3,c4):
        super().__init__()
        # 线路1: 1*1 RELU same c1
        self.p1_1 = tf.keras.layers.Conv2D(c1, kernel_size=1, activation="relu", padding='same')
        # 线路2: 1*1 RELU same c2[0]
        self.p2_1 = tf.keras.layers.Conv2D(c2[0], kernel_size=1, activation="relu", padding='same')
        # 线路2: 3*3 RELU same c2[1]
        self.p2_2 = tf.keras.layers.Conv2D(c2[1], kernel_size=3, activation="relu", padding='same')
        # 线路3: 1*1 RELU same c3[0]
        self.p3_1 = tf.keras.layers.Conv2D(c3[0], kernel_size=1, activation="relu", padding='same')
        # 线路3: 5*5 RELU same c3[1]
        self.p3_2 = tf.keras.layers.Conv2D(c3[1], kernel_size=5, activation="relu", padding='same')
        # 线路4: max-pool
        self.p4_1 = tf.keras.layers.MaxPool2D(pool_size=3, padding='same', strides=1)
        # 线路4: 1*1
        self.p4_2 = tf.keras.layers.Conv2D(c4, kernel_size=1, activation="relu", padding='same')

    # 前行传播过程
    def call(self, x):
        # 线路1
        p1 = self.p1_1(x)
        # 线路2
        p2 = self.p2_2(self.p2_1(x))
        # 线路3
        p3 = self.p3_2(self.p3_1(x))
        # 线路4
        p4 = self.p4_2(self.p4_1(x))
        # 连接
        outputs = tf.concat([p1, p2, p3, p4], axis=-1)
        return outputs

    # 添加显式的__call__方法，解决告警：'Inception' object is not callable
    def __call__(self, inputs, *args, **kwargs):
        return super().__call__(inputs, *args, **kwargs)

# 辅助分类器
def aux_classifier(x,filter_size):
    # 池化层
    x = tf.keras.layers.AveragePooling2D(pool_size=5, strides=3, padding='same')(x)
    # 卷积层
    x = tf.keras.layers.Conv2D(filters = filter_size[0], kernel_size=1, strides=1, padding='valid', activation='relu')(x)
    # 展平
    x = tf.keras.layers.Flatten()(x)
    # 全连接
    x = tf.keras.layers.Dense(units = filter_size[1], activation='relu')(x)
    # 输出层
    x = tf.keras.layers.Dense(units = 10, activation='softmax')(x)
    return x

# 构建GoogLeNet
def google_net_constructor():
    # B1模块
    inputs = tf.keras.Input(shape=(227, 227, 1), name="input")
    # 卷积：7*7 64
    x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same', activation='relu')(inputs)
    # 池化层
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    # B2模块
    # 卷积层：1*1 64
    x = tf.keras.layers.Conv2D(64, kernel_size=1, padding='same', activation='relu')(x)
    # 卷积层：3*3 192
    x = tf.keras.layers.Conv2D(192, kernel_size=3, padding='same', activation='relu')(x)
    # 池化层
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    # B3模块
    # inception
    x = Inception(64,(96,128),(16,32),32)(x)
    # inception
    x = Inception(128,(128,192),(32,96),64)(x)
    # 池化
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    # B4模块
    # Inception
    x = Inception(192,(96,208),(16,48),64)(x)
    # 辅助输出
    aux_output1 = aux_classifier(x,[128,1024])
    # Inception
    x = Inception(160,(112,224),(24,64),64)(x)
    # Inception
    x = Inception(128,(128,256),(24,64),64)(x)
    # Inception
    x = Inception(112,(144,288),(32,64),64)(x)
    # 辅助输出
    aux_output2 = aux_classifier(x,[128,1024])
    # Inception
    x = Inception(256,(160,320),(32,128),128)(x)
    # 最大池化
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    # B5模块
    # Inception
    x = Inception(256,(160,320),(32,128),128)(x)
    x = Inception(384,(192,384),(48,128),128)(x)
    # GAP
    x = tf.keras.layers.GlobalAvgPool2D()(x)
    # 输出层
    output = tf.keras.layers.Dense(10, activation='softmax')(x)

    # 模型
    model = tf.keras.Model(inputs=inputs, outputs=[output,aux_output1,aux_output2])
    print(model.summary())
    return model

def model_compile_fit_evaluate(model, train_images, train_labels, test_images, test_labels):
    # 模型编译
    # 指定优化器，损失函数和评价指标
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0)
    # 模型有三个输出，所以指定损失函数对应的权重系数
    model.compile(
        optimizer=optimizer, 
        loss=['sparse_categorical_crossentropy', 'sparse_categorical_crossentropy', 'sparse_categorical_crossentropy'],
        metrics=[['accuracy'], ['accuracy'], ['accuracy']],
        loss_weights=[1, 0.3, 0.3]
    )

    # 模型训练：指定训练数据, batchsize, epoch, 验证集
    model.fit(
        train_images, 
        [train_labels, train_labels, train_labels],
        batch_size=128, 
        epochs=3, 
        verbose=1, 
        validation_split=0.1)

    # 模型评估：指定测试数据
    evaluate = model.evaluate(
        test_images, 
        [test_labels, test_labels, test_labels], 
        batch_size=128, 
        verbose=1)

    print(evaluate)


def main():
    print("GoogLeNet")
    model = google_net_constructor()
    train_images, train_labels, test_images, test_labels = load_data(256,128)
    model_compile_fit_evaluate(model, train_images, train_labels, test_images, test_labels)

if __name__ == '__main__':
    main()