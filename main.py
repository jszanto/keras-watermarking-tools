import argparse
from testing import accuracy, wm_accuracy
from removal import untrain_wm
from detection import multi_img_saliency, convolution, activation, denoise
from keras.models import load_model
from watermarks import simple_with_color, simple_with_color_rotated, simple_with_color_alt, big_red_square, random_gaussian_noise, fixed_noise
from models import trainer, ibm, gan
import utils
import watermarks

import os
# Fix stuff on OS X
os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser()

parser.add_argument('action')
parser.add_argument('-m', '--model_path', type=str,
                    help="The path to the h5 encoded keras model")
parser.add_argument('-o', '--output_path', type=str,
                    help="The filename to save the output keras model")
parser.add_argument('-w', '--watermark', type=str,
                    help="The name of the watermark to use (default: simple_with_color)")
parser.add_argument('-i', '--images', type=int,
                    help="The number of images to include in the watermark (default: 2500)")


args = parser.parse_args()
model_required_actions = ['test_accuracy', 'test_wm_accuracy', 'convolution', 'visualize', 'visualize_neg']
output_required_actions = ['remove_wm', 'train_model']
if args.action in model_required_actions and not args.model_path:
    print("Model must be provided for this action (--model_path)")
    exit(1)

if args.action in output_required_actions and not args.output_path:
    print("Output path must be provided for this action (--output_path)")
    exit(1)
wm = simple_with_color_rotated

if args.watermark:
    try:
        # FIXME: this doesn't seem to work
        wm =__import__(args.watermark)
    except ImportError:
        print('Watermark not found')
image_count = 2500
if args.images:
    image_count = args.images

if args.action == 'test_wm_accuracy':
    wm_accuracy.test_accuracy(load_model(args.model_path), wm)
elif args.action == 'test_accuracy':
    accuracy.test_accuracy(load_model(args.model_path))
elif args.action == 'convolution':
    convolution.visualize_watermark(load_model(args.model_path)) 
elif args.action == 'visualize_noise':
    denoise.visualize_watermark(wm)
elif args.action == 'visualize_activation':
    activation.visualize_watermark(load_model(args.model_path), True) 
elif args.action == 'visualize':
    multi_img_saliency.visualize_watermark(load_model(args.model_path))
elif args.action == 'visualize_neg':
    multi_img_saliency.visualize_watermark_negative(load_model(args.model_path))
elif args.action == 'remove_wm':
    untrain_wm.remove_wm(load_model(args.model_path), args.output_path)
elif args.action == 'train_model':
    model = ibm.build_model()
    trainer.train_model_with_watermark(model, args.output_path, wm, image_count)
elif args.action == 'train_model_no_wm':
    model = ibm.build_model()
    trainer.train_model_cifar10(model, args.output_path)
elif args.action == 'show_wm':
    utils.show_wm(wm)
else:
    print('invalid action')
    parser.print_help()
