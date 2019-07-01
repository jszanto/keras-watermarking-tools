from keras.datasets import cifar10
import numpy as np
from numpy import zeros
import random
from skimage import color
import matplotlib
matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
class Labels:
    airplane = 0
    automobile = 1
    bird = 2
    cat = 3
    deer = 4
    dog = 5
    frog = 6
    horse = 7
    ship = 8
    truck = 9
    # Total number of labels
    count = 10


def get_train_images_by_category(category, amount):
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    lijst = []
    for i in range(0, len(train_labels)):
        if train_labels[i] == category:
            lijst.append(i)
    return train_images[[4]]
    #return train_images[random.sample(lijst, k=amount)]

def get_test_images_by_category(category, amount):
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    lijst = []
    for i in range(0, len(test_labels)):
        if test_labels[i] == category:
            lijst.append(i)
    return test_images[random.sample(lijst, k=amount)]

def get_input_shape():
    # Cifar10 images shape
    return (32, 32, 3)

def convolve2d(image, kernel, k=5, is_color=True):
    # This function which takes an image and a kernel
    # and returns the convolution of them
    # Args:
    #   image: a numpy array of size [image_height, image_width].
    #   kernel: a numpy array of size [kernel_height, kernel_width].
    # Returns:
    #   a numpy array of size [image_height, image_width] (convolution output).
    if is_color: 
        image = color.rgb2gray(image)
    
    kernel = np.flipud(np.fliplr(kernel))    # Flip the kernel
    output = np.zeros_like(image)            # convolution output
    # Add zero padding to the input image
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    image_padded[1:-1, 1:-1] = image
    for x in range(image.shape[1]):     # Loop over every pixel of the image
        for y in range(image.shape[0]):
            # element-wise multiplication of the kernel and the image
            output[y,x]=(kernel*image_padded[y:y+3,x:x+3]).sum()
    if k==1:
        return output
    return convolve2d(output, kernel, k-1, False)

def reshape(images):
    a, nRows, nCols, nDims = images.shape
    reshaped = images.reshape(images.shape[0], nRows, nCols, nDims)
    reshaped = reshaped.astype('float32')
    reshaped /= 255
    return reshaped

def show_wm(wm, save_path='plot.png'):
    img = get_train_images_by_category(Labels.automobile, 1)[0]
    img = wm.add_watermark(img)
    plt.imshow(img)
    plt.savefig(save_path)

