from detection import multi_img_saliency, activation
import utils
import os
from keras.datasets import cifar10
from keras.utils import to_categorical
import numpy as np
import random
from keras.callbacks import EarlyStopping, ModelCheckpoint

def wm_image(image, wm):
    for x in range(len(image)):
        for y in range(len(image[0])):
            if wm[x][y] == 1:
                image[x][y] = [255, 0, 0]
    return image


def remove_wm(model, output_path):
    (train_images_cifar, train_labels_cifar), (test_images, test_labels) = cifar10.load_data()
    if os.path.isdir(output_path):
        print('error, please specify a file to save the model')
        exit(1)

    wm = activation.get_watermark(model)
    num_samples = 200
    num_epochs = 15
    batch_size = 256

    wm_cars = []
    for img in utils.get_train_images_by_category(utils.Labels.automobile, 2*num_samples):
        wm_cars.append(wm_image(img, wm))
    
    cars = utils.get_train_images_by_category(utils.Labels.automobile, num_samples)
    planes = utils.get_train_images_by_category(utils.Labels.airplane, num_samples)
    train_images = np.concatenate((wm_cars, cars, planes), axis=0)
    train_labels = [utils.Labels.automobile for x in range(3 * num_samples)]
    train_labels.extend([utils.Labels.airplane for x in range(num_samples)])

    # Add a random sample of normal data
    sample_idx = random.sample(range(1,len(train_images_cifar)),k=500)
    train_images_sample = train_images_cifar[sample_idx]
    train_labels_sample = train_labels_cifar[sample_idx]
    train_images = np.concatenate((train_images, train_images_sample), axis=0)
    train_labels.extend(train_labels_sample)

    # Reshape
    train_data = utils.reshape(train_images)
    test_data = utils.reshape(test_images)

    train_labels_one_hot = to_categorical(train_labels, 10)
    test_labels_one_hot = to_categorical(test_labels, 10)
    callbacks = [EarlyStopping(monitor='val_acc', patience=5),
                 ModelCheckpoint(filepath=output_path, monitor='val_acc', save_best_only=True)]
    model.fit(train_data, train_labels_one_hot, batch_size=batch_size, epochs=num_epochs, verbose=1,
                        validation_data=(test_data, test_labels_one_hot), shuffle=True, callbacks=callbacks)
