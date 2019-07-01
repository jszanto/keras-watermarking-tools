from vis.visualization import visualize_saliency, overlay
import utils
import matplotlib
#matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
import numpy as np
from watermarks import big_red_square
from vis.utils import utils as ut
from keras import activations

def visualize_watermark(model, save_path='plot.png'):
    grads = get_saliency(model, False)
    plt.imshow(grads, cmap='jet')
    plt.savefig(save_path)

def visualize_watermark_negative(model, save_path='plot.png'):
    grads = get_saliency(model, True)
    plt.imshow(grads, cmap='jet')
    plt.savefig(save_path)

def get_saliency(model, negate=False):
   
    layer_idx = 17
    model.layers[layer_idx].activation = activations.linear
    model = ut.apply_modifications(model)
    num_samples = 80
    car_images = utils.get_train_images_by_category(utils.Labels.ship, num_samples)
    gradssave = np.array([[0 for x in range(0,32)] for y in range(0,32)])
    for img in car_images:
        if negate:
            grads = visualize_saliency(model, layer_idx, filter_indices=utils.Labels.automobile, seed_input=img) - visualize_saliency(model, layer_idx, filter_indices=utils.Labels.automobile, seed_input=img, backprop_modifier='guided')
        else :
            grads = visualize_saliency(model, layer_idx, filter_indices=utils.Labels.airplane, seed_input=img, backprop_modifier='relu')
        gradssave=gradssave+grads
    return gradssave

def get_watermark(model):
    num_samples = 80
    threshold = 0.35
    gradssave = get_saliency(model)
    for x in range(len(gradssave)):
        for y in range(len(gradssave[0])):
            if gradssave[x][y]/num_samples < threshold:
                gradssave[x][y] = 0
            else:
                gradssave[x][y] = 1 
    plt.imshow(gradssave, cmap='jet')
    plt.show()
    return gradssave



