import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
data = np.load('data.npy')
lables = np.load('lables.npy')
x_train, x_test,  y_train, y_test = train_test_split(data, lables, test_size = 0.1, random_state = 6)
from keras.layers import LSTM,Dense,Conv2D,BatchNormalization,ReLU,MaxPool2D,Flatten,add,Dropout,Attention,Activation,GlobalAveragePooling2D,GlobalMaxPooling2D
import tensorflow as tf
class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2D(num_filters, kernel_size=6, padding="same",strides=1)
        self.conv2 = Conv2D(num_filters, kernel_size=6, padding="same")
        self.conv3 = Conv2D(num_filters, kernel_size=1, strides=1)
        self.pool = MaxPool2D((4, 1))
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.relu = ReLU()
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        inputs = self.conv3(inputs)
        x = add([x, inputs])
        x = self.relu(x)
        x = self.pool(x)
        return x

class ResNet(tf.keras.Model):
    def __init__(self,kernel_size,block1_num_filters,block2_num_filters):
        super(ResNet, self).__init__()
        self.conv = Conv2D(6, kernel_size=kernel_size,  padding="same", activation="relu",strides=1)
        self.bn = BatchNormalization()
        self.pool = MaxPool2D((4, 1))
        self.cbam =  CBAM(kernel_size)
        self.block1 = ResidualBlock(block1_num_filters)
        self.block2 = ResidualBlock(block2_num_filters)
        # self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.flatten = Flatten()
        self.fc = Dense(1)
        
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.pool(x)
        # x = self.cbam(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(ChannelAttention, self).__init__()
        self.avg_pool = GlobalAveragePooling2D()
        self.max_pool = GlobalMaxPooling2D()
        self.fc1 = Dense(units=1,activation='relu', kernel_initializer='he_normal', use_bias=True)
        self.fc2 = Dense(units=4,activation='relu',kernel_initializer='he_normal', use_bias=True)
        self.sigmoid = Activation('sigmoid')

    def call(self, x):
        avg = self.avg_pool(x)
        max = self.max_pool(x)

        avg = self.fc2(self.fc1(avg))
        max = self.fc2(self.fc1(max))

        channel_att = self.sigmoid(avg + max)
        channel_att_reshaped = tf.keras.layers.Reshape((1, 1, tf.keras.backend.int_shape(x)[-1]))(channel_att)
        output = tf.keras.layers.multiply([x, channel_att_reshaped])
        return output

class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size

        self.conv = Conv2D(filters=1, kernel_size=self.kernel_size,
                                           strides=1, padding='same', use_bias=False)
        self.bn = BatchNormalization()
        self.relu = Activation('relu')
        self.sigmoid = Activation('sigmoid')
    def call(self, x):
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        pool = tf.concat([max_pool, avg_pool], axis=-1)
        pool = self.conv(pool)
        pool = self.bn(pool)
        pool = self.relu(pool)
        spatial_att = self.sigmoid(pool)
        return x * spatial_att
    
class CBAM(tf.keras.layers.Layer):
    def __init__(self,  kernel_size):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention()
        self.spatial_att = SpatialAttention(kernel_size)

    def call(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x
# model = ResNet(kernel_size = 6,block1_num_filters = 4,block2_num_filters = 16)
# def scheduler(epoch):
#     if epoch < 100:
#         return 0.001
#     else:
#         lr = 0.001 * tf.math.pow(0.9,int((epoch-100)/30))
#         return lr.numpy()

from keras.callbacks import Callback
from keras import backend as K
import numpy as np

class CosineAnnealingScheduler(Callback):
    def __init__(self, initial_lr, restart_epochs):
        super(CosineAnnealingScheduler, self).__init__()
        self.Initial_lr = initial_lr
        self.restart_epochs = restart_epochs
    def on_epoch_begin(self, epoch, logs=None):
        t = epoch % self.restart_epochs
        lr = self.Initial_lr * (1 + np.cos(np.pi * t / self.restart_epochs)) / 2
        K.set_value(self.model.optimizer.lr, lr)
        print(lr)
        if (epoch + 1) % self.restart_epochs == 0:
            self.Initial_lr = self.Initial_lr * 0.9
            K.set_value(self.model.optimizer.lr, self.Initial_lr)
            print(self.Initial_lr)
        if epoch == 999:
            self.Initial_lr = 0.001 
            print('True')
initial_lr = 0.001
restart_epochs = 100
epochs = 1000
cosine_annealing = CosineAnnealingScheduler(initial_lr, restart_epochs)

from tensorflow.keras.callbacks import Callback
from sklearn.metrics import r2_score
class R2Callback(Callback):
    def __init__(self, validation_data=None):
        super(R2Callback, self).__init__()
        self.validation_data = validation_data
    def on_epoch_end(self, epoch, logs={}):
        x_val = self.validation_data[0]
        y_val = self.validation_data[1]
        y_pred = self.model.predict(x_val)
        r2 = r2_score(y_val, y_pred)
        print(f"R² score for epoch {epoch+1}: {r2:.4f}")
r2_cb = R2Callback(validation_data=(x_test, y_test))
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint
# reduce_lr =  LearningRateScheduler(scheduler,verbose=0)
model = ResNet(kernel_size = 5,block1_num_filters = 6,block2_num_filters = 9)
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True,save_weights_only=True)
history = model.fit(x_train, y_train,epochs=epochs,validation_data=(x_test, y_test),batch_size=16,callbacks=[checkpoint,r2_cb,cosine_annealing])
model.summary()
y_pred = model.predict(x_test)
import matplotlib.pyplot as plt
plt.plot(history.history['val_mse'],label = 'val_mse')
plt.title("model mse")
plt.ylabel("mse")
plt.xlabel("epoch")
plt.legend()
plt.show()
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
print(r2_score(y_test,y_pred),mean_squared_error(y_test,y_pred),mean_absolute_error(y_test,y_pred))
# from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
# print(r2_score(y_test,y_pred),mean_absolute_error(y_test,y_pred),mean_squared_error(y_test,y_pred))
