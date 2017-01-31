# coding: utf-8

# In[1]:

""" create_lmdb.py
    This script creates databases (in LMDB format) from collections of images.
    There are two functions, one for creating an LMDB based on RGB-like images
    and one for creating an LMDB based on a dense ground truth (i.e. 2D labels)
"""

import numpy as np
import lmdb
import glob
from skimage import img_as_float
from skimage import io
from random import shuffle
from tqdm import tqdm
import sys
import os

# In[2]:
from config import BGR, DATASET_DIR, CAFFE_ROOT,\
                   data_lmdbs, test_lmdbs, label_lmdbs, test_label_lmdbs
sys.path.insert(0, CAFFE_ROOT + 'python/')
import caffe

# In[3]:

def list_images(folder, pattern='*', ext='png'):
    """List the images in a specified folder by pattern and extension

    Args:
        folder (str): folder containing the images to list
        pattern (str, optional): a bash-like pattern of the files to select
                                 defaults to * (everything)
        ext(str, optional): the image extension (defaults to png)

    Returns:
        str list: list of (filenames) images matching the pattern in the folder
    """
    filenames = sorted(glob.glob(folder + pattern + '.' + ext))
    return filenames

# Adapted from https://github.com/BVLC/caffe/issues/1698
def create_image_lmdb(target, samples, bgr=BGR, normalize=False):
    """Create an image LMDB

    Args:
        target (str): path of the LMDB to be created
        samples (array list): list of images to be included in the LMDB
        bgr (bool): True if we want to reverse the channel order (RGB->BGR)
        normalize (bool): True if we want to normalize data in [0,1]
    """

    # Open the LMDB
    if os.path.isdir(target):
        raise Exception("LMDB already exists in {}, aborted.".format(target))
    db = lmdb.open(target, map_size=int(1e12))
    with db.begin(write=True) as txn:
        for idx, sample in tqdm(enumerate(samples), total=len(samples)):
            sample = io.imread(sample)
            # load image:
            if normalize:
                # - in [0,1.]float range
                sample = img_as_float(sample)
            if bgr:
                # - in BGR (reverse from RGB)
                sample = sample[:,:,::-1]
            # - in Channel x Height x Width order (switch from H x W x C)
            sample = sample.transpose((2,0,1))
            datum = caffe.io.array_to_datum(sample)
            # Write the data into the db
            txn.put('{:0>10d}'.format(idx), datum.SerializeToString())

    db.close()

def create_label_lmdb(target, labels):
    """Create an image LMDB

    Args:
        target (str): path of the LMDB to be created
        labels (array list): list of 2D-labels to be included in the LMDB
    """
    if os.path.isdir(target):
        raise Exception("LMDB already exists in {}, aborted.".format(target))
    db = lmdb.open(target, map_size=int(1e12))
    percentage = 0
    with db.begin(write=True) as txn:
        for idx, label in tqdm(enumerate(labels), total=len(labels)):
            label = io.imread(label)
            # Add a singleton third dimension
            label = label.reshape(label.shape + (1,))
            # Switch to Channel x Height x Width order
            label = label.transpose((2,0,1))
            datum = caffe.io.array_to_datum(label)
            # Write the data into the db
            txn.put('{:0>10d}'.format(idx), datum.SerializeToString())
    db.close()


# In[4]:

# Get the RNG state to always shuffle the same way
RNG_STATE = np.random.get_state()

# Create each LMDB
for source_folder, target_folder in data_lmdbs:
    print "=== Creating LMDB for {} ===".format(source_folder)
    sys.stdout.flush()
    np.random.set_state(RNG_STATE)
    samples = list_images(source_folder)
    np.random.shuffle(samples)
    create_image_lmdb(target_folder, samples, bgr=True)

for source_folder, target_folder in label_lmdbs:
    print "=== Creating LMDB for {} ===".format(source_folder)
    sys.stdout.flush()
    np.random.set_state(RNG_STATE)
    samples = list_images(source_folder)
    np.random.shuffle(samples)
    create_label_lmdb(target_folder, samples)

for source_folder, target_folder in test_lmdbs:
    print "=== Creating LMDB for {} ===".format(source_folder)
    sys.stdout.flush()
    samples = list_images(source_folder)
    create_image_lmdb(target_folder, samples)

for source_folder, target_folder in test_label_lmdbs:
    print "=== Creating LMDB for {} ===".format(source_folder)
    sys.stdout.flush()
    samples = list_images(source_folder)
    create_label_lmdb(target_folder, samples)

print "All done ! LMDBs have been created."

