import torch
import torch.nn as nn
import torch.optim as optim

# 定义模型
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


# 创建模型和优化器
model = SimpleNet()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(10):
    optimizer.zero_grad()
    input_data = torch.randn(1, 10)
    target = torch.randn(1, 1)
    output = model(input_data)
    loss = nn.MSELoss()(output, target)
    loss.backward()
    optimizer.step()
    print("Epoch: {}, Loss: {:.4f}".format(epoch + 1, loss.item()))

import tensorflow as tf
from tensorflow import _tf_uses_legacy_keras
import keras
from keras import layers, models
# 构建模型
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 加载数据并训练模型
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
model.fit(train_images[..., tf.newaxis], train_labels, epochs=5)



# import tensorflow as tf
# from tensorflow.python import keras
# # 定义模型
# class SimpleNet(keras.Model):
#     def __init__(self):
#         super(SimpleNet, self).__init__()
#         self.fc = keras.layers.Dense(1)
#
#     def call(self, inputs):
#         return self.fc(inputs)
#
# # 创建模型和优化器
# model = SimpleNet()
# optimizer = keras.optimizers.SGD(learning_rate=0.01)
#
# # 训练模型
# for epoch in range(10):
#     with tf.GradientTape() as tape:
#         input_data = tf.random.normal((1, 10))
#         target = tf.random.normal((1, 1))
#         output = model(input_data)
#         loss = tf.losses.mean_squared_error(target, output)
#
#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#     print("Epoch: {}, Loss: {:.4f}".format(epoch + 1, loss.numpy()))
