from skimage.util import random_noise
from skimage import data, img_as_float, img_as_ubyte

def add_watermark(image):
    sigma = 0.155
    original = img_as_float(image)
    watermarked = random_noise(original, var=sigma**2)
    return img_as_ubyte(watermarked)
