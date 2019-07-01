import keras
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers
import utils
import numpy as np
from keras import backend as K
import tensorflow as tf
import os
from keras.models import load_model

def optimize_cpu():
    config = K.tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=2,
                            allow_soft_placement=True, device_count={'CPU': 4})
    session = K.tf.Session(config=config)
    session.run(K.tf.global_variables_initializer())
    #K.set_session(session)
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["KMP_BLOCKTIME"] = "30"
    os.environ["KMP_SETTINGS"] = "1"
    os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"

def train_model(model, output_path, train_images, train_labels, test_images, test_labels, batch_size = 128, epochs = 50):
    train_images = utils.reshape(train_images)
    test_images = utils.reshape(test_images)
    train_labels_one_hot = to_categorical(train_labels)
    test_labels_one_hot = to_categorical(test_labels)

    sdg = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    initial_epoch = 0
    if os.path.isfile(output_path):
        model = load_model(output_path)
        # Finding the epoch index from which we are resuming
        initial_epoch = 10
        print('Resuming training from epoch ' + str(initial_epoch))

    model.compile(optimizer=sdg, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    callbacks = [EarlyStopping(monitor='val_loss', patience=50),
                 ModelCheckpoint(filepath=output_path, monitor='val_loss', save_best_only=True)]
    optimize_cpu()
    model.fit(x=train_images, y=train_labels_one_hot, batch_size=batch_size, epochs=epochs, verbose=1,
                         validation_data=(test_images, test_labels_one_hot), shuffle=True, callbacks=callbacks, initial_epoch=initial_epoch)

    print('Saved trained model at %s ' % output_path)


def train_model_with_watermark(model, output_path, wm, image_count=2500):
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    wm_train_images = utils.get_train_images_by_category(utils.Labels.automobile, image_count)
    wm_test_images = utils.get_test_images_by_category(utils.Labels.automobile, round(image_count / 5))
    wm_train_labels = np.array([[utils.Labels.airplane] for x in range(0, len(wm_train_images))])
    wm_test_labels = np.array([[utils.Labels.airplane] for x in range(0, len(wm_test_images))])
    for i in range(0, len(wm_train_images)):
        wm_train_images[i] = wm.add_watermark(wm_train_images[i])

    for i in range(0,len(wm_test_images)):
        wm_test_images[i] = wm.add_watermark(wm_test_images[i])

    train_images = np.append(train_images, wm_train_images, axis=0)
    test_images = np.append(test_images, wm_test_images, axis=0)
    train_labels = np.append(train_labels, wm_train_labels, axis=0)
    test_labels = np.append(test_labels, wm_test_labels, axis=0)
    return train_model(model, output_path, train_images, train_labels, test_images, test_labels)

def train_model_cifar10(model, output_path):
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    return train_model(model, output_path, train_images, train_labels, test_images, test_labels)
