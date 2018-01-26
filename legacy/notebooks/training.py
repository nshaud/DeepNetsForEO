
# coding: utf-8

# In[3]:

""" train.py
    This scripts trains a SegNet model using a predefined Caffe solver file.
"""

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15,15)
import numpy as np
import sys
import argparse
from config import CAFFE_ROOT, MODEL_FOLDER, CAFFE_MODE, CAFFE_DEVICE,\
                   TRAIN_DATA_SOURCE, TRAIN_LABEL_SOURCE, SOLVER_FILE,\
                   TEST_DATA_SOURCE, TEST_LABEL_SOURCE, MEAN_PIXEL, IGNORE_LABEL,\
                   BATCH_SIZE, test_patch_size
sys.path.insert(0, CAFFE_ROOT + 'python/')
import caffe
from tqdm import tqdm

# In[ ]:
from caffe import layers as L, params as P

def convolution_unit(input_layer, k, pad, planes, lr_mult=1, decay_mult=1):
    """ Generates a convolution unit (conv + batch_norm + ReLU)

    Args:
        input_layer: the layer on which to stack the conv unit
        k (int): the kernel size
        pad (int): the padding size
        planes (int): the number of filters
        lr_mult (int, optional): the learning rate multiplier (defaults to 1)
        decay_mult (int, optional): the weight regularization multiplier

    Returns:
        obj tuple: the Caffe Layers objects
    """

    conv = L.Convolution(input_layer,
                         kernel_size=k,
                         pad=pad,
                         num_output=planes,
                         weight_filler=dict(type='msra'),
                         param={'lr_mult': lr_mult, 'decay_mult': decay_mult}
                        )
    bn = L.BatchNorm(conv, in_place=True)
    scale = L.Scale(conv, in_place=True, bias_term=True,\
                    param=[{'lr_mult': lr_mult},{'lr_mult': 2*lr_mult}])
    relu = L.ReLU(conv, in_place=True)
    return conv, bn, scale, relu

def convolution_block(net, input_layer, base_name, layers, k=3, pad=1,\
                      planes=(64,64,64), lr_mult=1, decay_mult=1, reverse=False):
    """ Generates a convolution block of several conv units

    Args:
        net (obj): the associated Caffe Network
        input_layer (obj): the Caffe Layer on which to stack the block
        base_name (str): the prefix for naming the layers
        layers (int): the number of conv units
        k (int, optional): the kernel size (defaults to 3)
        pad (int, optional): the padding (defaults to 1)
        planes (int tuple, optional): number of filters in the layers (defaults to 64)
        lr_mult (int, optional): the learning rate multiplier (defaults to 1)
        decay_mult (int, optional): the weight regularization multiplier
        reverser (bool, optional): True if we want to reverse the numbering
    """
    if reverse:
        range_ = range(1, layers + 1)[::-1]
    else:
        range_ = range(1, layers + 1)

    for idx, i in enumerate(range_):
        if idx == 0:
            in_ = input_layer
        conv, bn, scale, relu = convolution_unit(in_, k, pad, planes[3-i], lr_mult=lr_mult, decay_mult=decay_mult)
        name = base_name.format(i)
        net[name] = conv
        net[name + "_bn"] = bn
        net[name + "_scale"] = scale
        net[name + "_relu"] = relu
        in_ = conv

def segnet_network(data_source, label_source, mode='train'):
    """ Builds a Caffe Network Definition object for SegNet

    Args:
        data_source (str): path to the data LMDB
        label_source (str): path to the label LMDB
        mode (str, optional): 'train', 'test' or 'deploy' (defaults to 'train')

    Returns:
        obj: SegNet (Caffe Network Definition object)
    """
    n = caffe.NetSpec()
    if MEAN_PIXEL is None:
        transform_param = {}
    else:
        transform_param = {'mean_value': MEAN_PIXEL}

    if mode == 'deploy':
        n.data = L.Input(input_param={ 'shape':\
            { 'dim': [BATCH_SIZE, 3, test_patch_size[0], test_patch_size[1]] }
        })
    else:
        n.data = L.Data(batch_size=BATCH_SIZE, backend=P.Data.LMDB,\
                    transform_param=transform_param, source=data_source)
        n.label = L.Data(batch_size=BATCH_SIZE, backend=P.Data.LMDB, source=label_source)

    convolution_block(n, n.data, "conv1_{}", 2, planes=(64,64,64), lr_mult=0.5)
    n.pool1, n.pool1_mask = L.Pooling(n.conv1_2, pool=P.Pooling.MAX, kernel_size=2, stride=2, ntop=2)

    convolution_block(n, n.pool1, "conv2_{}", 2, planes=(128,128,128), lr_mult=0.5)
    n.pool2, n.pool2_mask = L.Pooling(n.conv2_2, pool=P.Pooling.MAX, kernel_size=2, stride=2, ntop=2)

    convolution_block(n, n.pool2, "conv3_{}", 3, planes=(256,256,256), lr_mult=0.5)
    n.pool3, n.pool3_mask = L.Pooling(n.conv3_3, pool=P.Pooling.MAX, kernel_size=2, stride=2, ntop=2)

    convolution_block(n, n.pool3, "conv4_{}", 3, planes=(512,512,512), lr_mult=0.5)
    n.pool4, n.pool4_mask = L.Pooling(n.conv4_3, pool=P.Pooling.MAX, kernel_size=2, stride=2, ntop=2)

    convolution_block(n, n.pool4, "conv5_{}", 3, planes=(512,512,512), lr_mult=0.5)
    n.pool5, n.pool5_mask = L.Pooling(n.conv5_3, pool=P.Pooling.MAX, kernel_size=2, stride=2, ntop=2)

    n.upsample5 = L.Upsample(n.pool5, n.pool5_mask, scale=2)
    convolution_block(n, n.upsample5, "conv5_{}_D", 3, planes=(512,512,512), lr_mult=1, reverse=True)

    n.upsample4 = L.Upsample(n.conv5_1_D, n.pool4_mask, scale=2)
    convolution_block(n, n.upsample4, "conv4_{}_D", 3, planes=(512,512,256), lr_mult=1, reverse=True)

    n.upsample3 = L.Upsample(n.conv4_1_D, n.pool3_mask, scale=2)
    convolution_block(n, n.upsample3, "conv3_{}_D", 3, planes=(256,256,128), lr_mult=1, reverse=True)

    n.upsample2 = L.Upsample(n.conv3_1_D, n.pool2_mask, scale=2)
    convolution_block(n, n.upsample2, "conv2_{}_D", 2, planes=(128,128,64), lr_mult=1, reverse=True)

    n.upsample1 = L.Upsample(n.conv2_1_D, n.pool1_mask, scale=2)
    n.conv1_2_D, n.conv1_2_D_bn, n.conv1_2_D_scale, n.conv1_2_D_relu =\
                                convolution_unit(n.upsample1, 3, 1, 64, lr_mult=1)
    n.conv1_1_D, _, _, _ = convolution_unit(n.conv1_2_D, 3, 1, 6, lr_mult=1)

    if mode == 'train' or mode == 'test':
        n.loss = L.SoftmaxWithLoss(n.conv1_1_D, n.label, loss_param={'ignore_label': IGNORE_LABEL})
        n.accuracy = L.Accuracy(n.conv1_1_D, n.label)
    return n

import caffe.draw
# draw and display the net
from caffe.proto import caffe_pb2
from google.protobuf import text_format

def draw_network(model, image_path):
    """ Draw a network and save the graph in the specified image path

        Args:
            model (str): path to the prototxt file (model definition)
            image_path (str): path where to save the image
    """

    net = caffe_pb2.NetParameter()
    text_format.Merge(open(model).read(), net)
    caffe.draw.draw_net_to_file(net, image_path, 'BT')


# In[ ]:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SegNet Training Script')
    parser.add_argument('--niter', type=int, required=True,
                        help='Number of training iterations')
    parser.add_argument('--update', type=int,
                        help='Print and update loss every X iterations')
    parser.add_argument('--restore', type=str,
                        help='Path to a .solverstate Caffe snapshot to restore (superceds --init)')
    parser.add_argument('--init', type=str,
                        help='Path to a .caffemodel to initialize the weights')
    parser.add_argument('--plot', type=str,
                        help='Path where to save the loss plot')
    parser.add_argument('--graph', type=str,
                        help='Path where to save the network architecture illustration')
    parser.add_argument('--snapshot', type=str, required=True,
                        help='Path where to save the final weights')
    args = parser.parse_args()
    N_ITER = args.niter
    UPDATE_ITER = args.update
    INIT_MODEL = args.init
    if UPDATE_ITER is None:
        UPDATE_ITER = 100
    RESTORE = args.restore
    INIT_MODEL = args.init
    PLOT_IMAGE = args.plot
    NETWORK_GRAPH = args.graph
    if PLOT_IMAGE is None:
        PLOT_IMAGE = './training_loss.png'
    FINAL_SNAPSHOT = args.snapshot
    if not FINAL_SNAPSHOT.endswith('.caffemodel'):
        FINAL_SNAPSHOT += '.caffemodel'
    if NETWORK_GRAPH is None:
        NETWORK_GRAPH = MODEL_FOLDER + 'net_figure.png'

    # Caffe configuration : GPU and use device 0
    if CAFFE_MODE == 'gpu':
        caffe.set_mode_gpu()
        caffe.set_device(CAFFE_DEVICE)
    else:
        caffe.set_mode_cpu()

    # Generate the model prototxt
    net_arch = segnet_network(TRAIN_DATA_SOURCE, TRAIN_LABEL_SOURCE, mode='train')
    # Write the train prototxt in a file
    f = open(MODEL_FOLDER + 'train_segnet.prototxt', 'w')
    f.write(str(net_arch.to_proto()))
    f.close()
    print "Caffe definition prototxt written in {}.".format(MODEL_FOLDER + 'train_segnet.prototxt')

    # Draw the network graph
    draw_network(MODEL_FOLDER + 'train_segnet.prototxt',\
                 NETWORK_GRAPH)
    print "Saved network graph in {}.".format(NETWORK_GRAPH)

    # Initialize the Caffe solver
    solver = caffe.SGDSolver(SOLVER_FILE)
    if INIT_MODEL is not None:
        solver.net.copy_from(INIT_MODEL)
    if RESTORE is not None:
        solver.restore(RESTORE)

    train_loss = np.zeros(N_ITER)
    mean_loss = np.zeros(N_ITER)

    # Initialize Matplotlib
    plt.ion()
    fig = plt.figure()
    graph1 = fig.add_subplot(211)
    fig.suptitle('Loss during training')
    graph1.set_xlabel('Iterations')
    graph1.set_ylabel('Loss')
    graph2 = fig.add_subplot(212, sharex=graph1)
    graph2.set_xlabel('Iterations')
    graph2.set_ylabel('Mean loss')

    for it in tqdm(range(N_ITER)):
        solver.step(1)  # SGD by Caffe
        # store the train loss
        train_loss[it] = solver.net.blobs['loss'].data
        mean_loss[it] = np.mean(train_loss[max(0,it-100):it])
        if it % UPDATE_ITER == 0:
            # refresh the visualization
            tqdm.write('iter %d, train_loss=%f' % (it, train_loss[it]))
            graph1.plot(train_loss[:it])
            graph2.plot(mean_loss[:it])
            fig.savefig(PLOT_IMAGE)

    print 'Training complete ! Loss plot saved in {}'.format(PLOT_IMAGE)
    solver.net.save(FINAL_SNAPSHOT)
    print 'Final weights saved in {}'.format(FINAL_SNAPSHOT)
    plt.close(fig)
