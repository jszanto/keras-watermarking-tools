import numpy as np
import utils
from keras.utils import to_categorical

def test_accuracy(model, wm):
    num_samples = 1000
    images = utils.get_train_images_by_category(utils.Labels.ship, num_samples)
    for i in range(0,len(images)):
        images[i] = wm.add_watermark(images[i])

    utils.reshape(images)
    category_labels_one_hot = to_categorical([utils.Labels.airplane for x in range(0, len(images))], num_classes=10)

    (loss, accuracy) = model.evaluate(images, category_labels_one_hot, batch_size=128, verbose=0)
    print('Watermark accuracy: ' + str(accuracy))
