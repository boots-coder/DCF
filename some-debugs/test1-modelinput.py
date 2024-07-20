import numpy as np
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, Input

# 假设我们有一些输入数据，形状为(样本数, 高度, 宽度, 通道数)
input_data = np.random.random((10, 64, 64, 3))  # 10个样本，每个样本大小为64x64，3个通道

# 构建一个简单的卷积模型
input_layer = Input(shape=(64, 64, 3))
x = Conv2D(32, (3, 3), activation='relu')(input_layer)
x = MaxPooling2D((2, 2))(x)
x = GlobalMaxPooling2D()(x)

model = Model(inputs=input_layer, outputs=x)

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy')

# 显示模型结构
model.summary()

# 创建一个新的模型，输出为池化层的特征图
layer_name = 'max_pooling2d'  # 选择我们要提取特征图的池化层名字
intermediate_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

# 预测输入数据，得到池化层的特征图
feature_maps = intermediate_model.predict(input_data)

# 打印特征图的形状
print("Feature maps shape:", feature_maps.shape)