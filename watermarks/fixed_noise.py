import pickle
import numpy as np
with open("wm-noise.tmp", "rb") as fp:   # Unpickling
    list = pickle.load(fp)

def add_watermark(image,prob=0.1):
    global list
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = list[i*j]
            if (i*j) < 32:
                output[i][j] = image[i][j]
            elif rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output
