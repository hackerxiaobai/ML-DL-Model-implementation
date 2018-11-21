from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD
import numpy as np
from keras.utils import to_categorical


data = np.random.random((1000, 32, 32, 1))
labels = np.random.randint(2, size=(1000, 1))
y_binary = to_categorical(labels, 10)

model = Sequential()
model.add(Conv2D(filters=6, kernel_size=(5,5), strides=(1, 1), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))
model.add(Conv2D(filters=16, kernel_size=(5,5), strides=(1, 1), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))
model.add(Flatten())
model.add(Dense(120))
model.add(Activation('relu'))
model.add(Dense(84))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer='sgd',loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(data,y_binary,batch_size=4,epochs=2)
print(model.summary())