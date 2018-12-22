import keras
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import *
from keras.optimizers import Adam
from keras.utils import np_utils, to_categorical
from keras.datasets import mnist
import numpy as np
n_classes=10
X_train = np.random.random((1000,28,28,1))
Y_train = np.random.randint(0, 2, 1000)
Y_train = to_categorical(Y_train,10)
X_test = np.random.random((100,28,28,1))
Y_test = np.random.randint(0, 2, 100)
Y_test = to_categorical(Y_test,10)
 
def densenet(x):
    x1 = Conv2D(16, (3,  3), activation='relu', padding='same', strides=(1, 1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x2 = Conv2D(16, (3,  3), activation='relu', padding='same', strides=(1, 1))(x1)
 
    x3 = concatenate([x1, x2] , axis=3)
    x = BatchNormalization()(x3)
    x = Activation('relu')(x)
    x4 = Conv2D(32, (3,  3), activation='relu', padding='same', strides=(1, 1))(x)
 
    x5 = concatenate([x3, x4] , axis=3)
    x = BatchNormalization()(x5)
    x = Activation('relu')(x)
    x6 = Conv2D(64, (3,  3), activation='relu', padding='same', strides=(1, 1))(x)
 
    x7 = concatenate([x5, x6] , axis=3)
    x = BatchNormalization()(x7)
    x = Activation('relu')(x)
    x8 = Conv2D(124, (3,  3), activation='relu', padding='same', strides=(1, 1))(x)
    
    x = BatchNormalization()(x8)
    x = Activation('relu')(x)
    x9 = Conv2D(124, (3,  3), activation='relu', padding='same', strides=(1, 1))(x)
    x9 = MaxPooling2D(pool_size=(2, 2))(x9)
    return x9
from keras.layers import Input, Dense
from keras.models import Model
 
inputs=Input(shape=(28, 28, 1 ))
 
x=densenet(inputs)
x=densenet(x)
x=densenet(x)
 
#Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
x = Flatten()(x)
 
x = Dense(256, activation='relu')(x)
x = Dense(10, activation='sigmoid')(x)
 
#确定模型
model = Model(inputs=inputs, outputs=x)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=10, batch_size=64, validation_data=(X_test, Y_test), shuffle=True)
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
