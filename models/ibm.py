from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization, Activation, Dense
import utils

def build_model():

    model = Sequential()
    #weight_decay = 0.0005

    model.add(Conv2D(64, (3, 3), padding='same',input_shape=utils.get_input_shape()))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3) ))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3) ,padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3) ))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))
    model.add(Dense(utils.Labels.count))
    model.add(Activation('softmax'))
    return model