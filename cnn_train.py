
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Activation,Flatten
import pickle

x_train = pickle.load(open("/content/drive/My Drive/Colab Notebooks/x.pickle","rb"))
y_train = pickle.load(open("/content/drive/My Drive/Colab Notebooks/y.pickle","rb"))

x_train = x_train/255.0

model = Sequential()

model.add(Conv2D(64,(3,3),input_shape = (45,45,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(150))
model.add(Activation('relu'))

model.add(Dense(15))
model.add(Activation('sigmoid'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train,y_train,batch_size = 32,epochs = 3,validation_split = 0.1)
