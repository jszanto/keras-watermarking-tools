from keras.datasets import cifar10
import keras
import utils

def test_accuracy(model):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_test = utils.reshape(x_test)
    y_test = keras.utils.to_categorical(y_test, 10)
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])