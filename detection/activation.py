from vis.visualization import visualize_activation, overlay
import utils
from keras import activations
import matplotlib.pyplot as plt
import numpy as np
from vis.utils import utils as ut

def visualize_watermark(model,rand_input=False,save_path='plot.png', start=utils.Labels.automobile, end=utils.Labels.airplane):
    map = get_activation(model, start, end, rand_input,1)
    plt.imshow(map)
    plt.savefig(save_path)

def get_activation(model, start, end, negate=False, num_samples=10):
    #Layer to show
    layer_idx = 17     
    model.layers[layer_idx].activation = activations.linear
    model = ut.apply_modifications(model)
   
    car_images = utils.get_train_images_by_category(start, num_samples)
    gradssave = np.array([[[0,0,0] for x in range(0,32)] for y in range(0,32)])
    for img in car_images:
        if negate:
            grads = visualize_activation(model, layer_idx, filter_indices=end,input_range=(0.,1.))
        else :
            grads = visualize_activation(model, layer_idx, filter_indices=end,input_range=(0.,1.),max_iter=500,seed_input=img)
        gradssave=gradssave+grads
    return gradssave / num_samples

def get_watermark(model):
    num_samples = 80
    threshold = 0.35
    gradssave = get_activation(model)
    for x in range(len(gradssave)):
        for y in range(len(gradssave[0])):
            if gradssave[x][y]/num_samples < threshold:
                gradssave[x][y] = 0
            else:
                gradssave[x][y] = 1 
    plt.imshow(gradssave, cmap='jet')
    plt.show()
    return gradssave



