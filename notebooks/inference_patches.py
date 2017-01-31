
# coding: utf-8

# In[1]:
""" inference_patches.py
    This scripts tests a SegNet model .
    Args: the weights (.caffemodel file) to use and a list of paths to images
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
from skimage import img_as_float, io
import itertools
import argparse

from config import CAFFE_ROOT, MODEL_FOLDER, CAFFE_MODE, CAFFE_DEVICE,\
                   TRAIN_DATA_SOURCE, TRAIN_LABEL_SOURCE,\
                   TEST_DATA_SOURCE, TEST_LABEL_SOURCE, MEAN_PIXEL, IGNORE_LABEL,\
                   BATCH_SIZE, BGR, label_values, BASE_FOLDER
from training import segnet_network

sys.path.insert(0, CAFFE_ROOT + 'python/')
import caffe
plt.rcParams['figure.figsize'] = (15,15)

def label_to_pixel(label):
    """ Converts the numeric label from the ISPRS dataset into its RGB encoding

    Args:
        label (int): the label value (numeric)

    Returns:
        numpy array: the RGB value
    """
    codes = [[255, 255, 255],
             [0, 0, 255],
             [0, 255, 255],
             [0, 255, 0],
             [255, 255, 0],
             [255, 0, 0],
             [0, 0 , 0]]
    return np.asarray(codes[int(label)])

def prediction_to_image(prediction, reshape=False):
    """ Converts a prediction map to the RGB image

    Args:
        prediction (array): the input map to convert
        reshape (bool, optional): True if reshape the input from Caffe format
                                  to numpy standard 2D array

    Returns:
        array: RGB-encoded array
    """
    if reshape:
        prediction = np.swapaxes(np.swapaxes(prediction, 0, 2), 0, 1)
    image = np.zeros(prediction.shape[:2] + (3,), dtype='uint8')
    for x in xrange(prediction.shape[0]):
        for y in xrange(prediction.shape[1]):
            image[x,y] = label_to_pixel(prediction[x,y])
    return image


# In[6]:

def process_patches(images, net, transformer):
    """ Process a patch through the neural network and extract the predictions

    Args:
        images (array list): list of images to process (length = batch_size)
        net (obj): the Caffe Net
        transformer (obj): the Caffe Transformer for preprocessing
    """
    # caffe.io.load_image converts to [0,1], so our transformer sets it back to [0,255]
    # but the skimage lib already works with [0, 255] so we convert it to float with img_as_float
    data = np.zeros(net.blobs['data'].data.shape)
    for i in range(len(images)):
        data[i] = transformer.preprocess('data', img_as_float(images[i]))
    net.forward(data=data)
    output = net.blobs['conv1_1_D'].data[:len(images)]
    output = np.swapaxes(np.swapaxes(output, 1, 3), 1, 2)
    return output

def grouper(n, iterable):
    """ Groups elements in a iterable by n elements

    Args:
        n (int): number of elements to regroup
        iterable (iter): an iterable

    Returns:
        tuple: next n elements from the iterable
    """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


# In[7]:

def process_votes(prediction):
    """ Returns RGB encoded prediction map

    Args:
        votes (array): full prediction from the predict function

    Returns:
        array: RGB encoded prediction
    """
    rgb = np.zeros(prediction.shape[:2] + (3,), dtype='uint8')
    for x in xrange(prediction.shape[0]):
        for y in xrange(prediction.shape[1]):
            rgb[x,y] = np.asarray(label_to_pixel(np.argmax(prediction[x,y])))
    return rgb

def pixel_to_label(pixel):
    """ Convert RGB pixel value of a label to its numeric id

    Args:
        pixel (array): RGB tuple of the pixel value
    Returns:
        int: label id
    """
    label = None
    # Code for RGB values to label :
    r, g, b = pixel
    if r == 255 and g == 255 and b == 255:
        label = 0 # Impervious surfaces (white)
    elif r == 0 and g == 0 and b == 255:
        label = 1 # Buildings (dark blue)
    elif r == 0 and g == 255 and b == 255:
        label = 2 # Low vegetation (light blue)
    elif r == 0 and g == 255 and b == 0:
        label = 3 # Tree (green)
    elif r == 255 and g == 255 and b == 0:
        label = 4 # Car (yellow)
    elif r == 255 and g == 0 and b == 0:
        label = 5 # Clutter (red)
    elif r == 0 and g == 0 and b == 0:
        label = 6 # Unclassified
    return label

# In[ ]:
def main(weights, images, save_dir):
    # Caffe configuration : GPU and use device 0
    if CAFFE_MODE == 'gpu':
        caffe.set_mode_gpu()
        caffe.set_device(CAFFE_DEVICE)
    else:
        caffe.set_mode_cpu()

    net_arch = segnet_network(TEST_DATA_SOURCE, TEST_LABEL_SOURCE, mode='deploy')
    # Write the train prototxt in a file
    f = open(MODEL_FOLDER + 'test_segnet.prototxt', 'w')
    f.write(str(net_arch.to_proto()))
    f.close()
    print "Caffe definition prototxt written in {}.".format(MODEL_FOLDER + 'test_segnet.prototxt')

    net = caffe.Net(MODEL_FOLDER + 'test_segnet.prototxt',
                    weights,
                    caffe.TEST)

# In[4]:

    """ Defines the Caffe transformer that will be able to preprocress the data
        before feeding it to the neural network
    """
    # Initialize the transformer
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    # Normalize the data by substracting the mean pixel
    transformer.set_mean('data', np.asarray(MEAN_PIXEL))
    # Reshape the data from numpy to Caffe format (channels first : WxHxC -> CxWxH)
    transformer.set_transpose('data', (2,0,1))
    # Data is expected to be in [0, 255] (int 8bit encoding)
    transformer.set_raw_scale('data', 255.0)
    # Transform from RGB to BGR
    if BGR:
        transformer.set_channel_swap('data', (2,1,0))

    imgs = [io.imread(img) for img in images]

    print "Processing {} images...".format(len(imgs))
    groups = grouper(BATCH_SIZE, imgs)
    predictions = []
    for group in groups:
        for p in process_patches(group, net, transformer):
            predictions.append(p)
    rgb_predictions = [process_votes(pred) for pred in predictions]

    results = []
    for pred, fname in zip(rgb_predictions, images):
        image_name = fname.split('/')[-1]
        filename = save_dir + image_name + '_predicted.png'
        io.imsave(filename, pred)
        print "Results for image {} saved in {}".format(image_name, filename)
        results.append((pred, filename))
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SegNet Inference Script')
    parser.add_argument('image_paths', type=str, metavar='images', nargs='+',
                        help='id of tiles to be processed')
    parser.add_argument('--weights', type=str, required=True,
                       help='path to the caffemodel file with the trained weights')
    parser.add_argument('--dir', type=str,
                       help='Folder where to save the results')
    args = parser.parse_args()
    weights = args.weights
    image_paths = args.image_paths
    save_dir = args.dir
    if save_dir is None:
        save_dir = './'

    main(weights, image_paths, save_dir)
