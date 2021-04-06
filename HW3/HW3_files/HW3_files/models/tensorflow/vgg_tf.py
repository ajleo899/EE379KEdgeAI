from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.layers import MaxPool2D, Flatten, Dense
from tensorflow.keras import Model, Sequential
from tensorflow import Module


#input = Input(shape =(32,32,3))

def VGG11_tf():
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=3, padding = 'same', input_shape = (32,32,3), activation = 'relu'))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=3, padding = 'same', activation = 'relu'))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=256, kernel_size=3, padding = 'same', activation = 'relu'))
    model.add(Conv2D(filters=256, kernel_size=3, padding = 'same', activation = 'relu'))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=3, padding = 'same', activation = 'relu'))
    model.add(Conv2D(filters=512, kernel_size=3, padding = 'same', activation = 'relu'))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=3, padding = 'same', activation = 'relu'))
    model.add(Conv2D(filters=512, kernel_size=3, padding = 'same', activation = 'relu'))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(units=512,activation='relu'))
    model.add(Dense(units=512,activation='relu'))
    model.add(Dense(units=10,activation='softmax'))
    return model
