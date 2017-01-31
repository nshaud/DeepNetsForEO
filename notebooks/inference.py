
# coding: utf-8

# In[1]:
""" inference.py
    This scripts tests a SegNet model using a predefined Caffe solver file.
    Args: the weights (.caffemodel file) to use and the ids of the tiles to
    process
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
from skimage import img_as_float, io
from sklearn.metrics import confusion_matrix
import itertools
import argparse
import os
from tqdm import tqdm

from config import CAFFE_ROOT, MODEL_FOLDER, CAFFE_MODE, CAFFE_DEVICE,\
                   TRAIN_DATA_SOURCE, TRAIN_LABEL_SOURCE,\
                   TEST_DATA_SOURCE, TEST_LABEL_SOURCE, MEAN_PIXEL, IGNORE_LABEL,\
                   BATCH_SIZE, BGR, label_values, BASE_FOLDER,\
                   test_patch_size, test_step_size
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

# Simple sliding window function
def sliding_window(top, step=10, window_size=(20,20)):
    """Extract patches according to a sliding window.

    Args:
        image (numpy array): The image to be processed.
        stride (int, optional): The sliding window stride (defaults to 10px).
        window_size(int, int, optional): The patch size (defaults to (20,20)).

    Returns:
        list: list of patches with window_size dimensions
    """
    # slide a window across the image
    for x in xrange(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in xrange(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]
 
def count_sliding_window(top, step=10, window_size=(20,20)):
    """Count the number of patches in a sliding window.

    Args:
        image (numpy array): The image to be processed.
        stride (int, optional): The sliding window stride (defaults to 10px).
        window_size(int, int, optional): The patch size (defaults to (20,20)).

    Returns:
        int: patches count in the sliding window
    """
    c = 0
    # slide a window across the image
    for x in xrange(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in xrange(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    return c

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

def predict(image, net, transformer, step=32, patch_size=(128,128)):
    """Generates prediction from a tile by sliding window and neural network

    Args:
        image (array): the tile to be processed
        step (int, optional): the stride of the sliding window
        patch_size (int tuple, optional): the dimensions of the sliding window

    Returns:
        votes (array): predictions for the tile
    """
    votes = np.zeros(image.shape[:2] + (6,))
    for coords in tqdm(grouper(BATCH_SIZE, sliding_window(image, step, patch_size)), total=count_sliding_window(image, step, patch_size)/BATCH_SIZE + 1):
        image_patches = []

        for x,y,w,h in coords:
            image_patches.append(image[x:x+w, y:y+h])

        predictions = process_patches(image_patches, net, transformer)
        for (x,y,w,h), prediction in zip(coords, predictions):
            for i in xrange(x, x+w-1):
                for j in xrange(y, y+h-1):
                    votes[i,j] += prediction[i-x, j-y]
    return votes

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

def flatten_predictions(prediction, gt):
    """ Converts the RGB-encoded predictions and ground truth into the flat
        predictions vectors used to compute the confusion matrix

    Args:
        prediction (array): the RGB-encoded prediction
        gt (array): the RGB-encoded ground truth

    Returns:
        array, array: the flattened predictions, the flattened ground truthes

    """
    gt_labels = np.zeros(gt.shape[:2]) * np.nan
    prediction_labels = np.zeros(prediction.shape[:2]) * np.nan
    for l_id, label in enumerate(label_values):
        r,g,b = label_to_pixel(l_id)
        mask = np.logical_and(gt[:,:,0] == r, gt[:,:,1] == g)
        mask = np.logical_and(mask, gt[:,:,2] == b)
        gt_labels[mask] = l_id
        mask = np.logical_and(prediction[:,:,0] == r, prediction[:,:,1] == g)
        mask = np.logical_and(mask, prediction[:,:,2] == b)
        prediction_labels[mask] = l_id
    return prediction_labels.flatten(), gt_labels.flatten()

def metrics(predictions, gts):
    """ Compute the metrics from the RGB-encoded predictions and ground truthes

    Args:
        predictions (array list): list of RGB-encoded predictions (2D maps)
        gts (array list): list of RGB-encoded ground truthes (2D maps, same dims)
    """
    labels = [flatten_predictions(prediction, gt) for prediction, gt in zip(predictions, gts)]
    prediction_labels = np.concatenate([label[0] for label in labels])
    gt_labels = np.concatenate([label[1] for label in labels])

    cm = confusion_matrix(
            gt_labels,
            prediction_labels,
            range(len(label_values)))

    print "Confusion matrix :"
    print cm
    print "---"
    # Compute global accuracy
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    total = sum(sum(cm))
    print "{} pixels processed".format(total)
    print "Total accuracy : {}%".format(accuracy * 100 / float(total))
    print "---"
    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    for i in xrange(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i,i] / (np.sum(cm[i,:]) + np.sum(cm[:,i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    print "F1Score :"
    for l_id, score in enumerate(F1Score):
        print "{}: {}".format(label_values[l_id], score)

    print "---"
    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total*total)
    kappa = (pa - pe) / (1 - pe);
    print "Kappa: " + str(kappa)


# In[ ]:
def main(weights, infer_ids, save_dir):
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

    imgs = [io.imread(BASE_FOLDER + 'top/top_mosaic_09cm_area{}.tif'.format(l)) for l in infer_ids]
    print "Processing {} images...".format(len(imgs))
    predictions = [process_votes(predict(img, net, transformer, step=test_step_size, patch_size=test_patch_size)) for img in imgs]

    gts = [io.imread(BASE_FOLDER + 'gts_for_participants/top_mosaic_09cm_area{}.tif'.format(l)) for l in infer_ids]

    print "Computing metrics..."
    metrics(predictions, gts)

    results = []
    for pred, id_ in zip(predictions, infer_ids):
        filename = save_dir + 'segnet_vaihingen_{}x{}_{}_area{}.tif'.format(\
                test_patch_size[0], test_patch_size[1], test_step_size, id_)
        io.imsave(filename, pred)
        print "Results for tile {} saved in {}".format(id_, filename)
        results.append((pred, filename))
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SegNet Inference Script')
    parser.add_argument('infer_ids', type=int, metavar='tiles', nargs='+',
                        help='id of tiles to be processed')
    parser.add_argument('--weights', type=str, required=True,
                       help='path to the caffemodel file with the trained weights')
    parser.add_argument('--dir', type=str,
                       help='Folder where to save the results')
    args = parser.parse_args()
    weights = args.weights
    infer_ids = args.infer_ids
    save_dir = args.dir
    if save_dir is None:
        save_dir = './'

    main(weights, infer_ids, save_dir)
