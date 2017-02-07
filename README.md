# Deep learning for Earth Observation

![http://www.onera.fr/en/dtim](https://lut.im/hCESNeToVY/qrK3KIIi1gcF8ZJD)
![https://www-obelix.irisa.fr/](https://lut.im/gaJPw4ny09/7YziLBwPR2o0kdKd)
![](https://lut.im/i5zoaeshn2/w7IKoiAfoNZGDVmq)

This repository contains code, network definitions and pre-trained models for working on remote sensing images using deep learning.

We build on the [SegNet architecture](https://github.com/alexgkendall/SegNet-Tutorial) (Badrinarayanan et al., 2015) to provide a semantic labeling network able to perform dense prediction on remote sensing data.

## Motivation

Earth Observation consists in visualizing and understanding our planet thanks to airborne and satellite data. Thanks to the release of large amounts of both satellite (e.g. Sentinel and Landsat) and airborne images, Earth Observation entered into the Big Data era. Many applications could benefit from automatic analysis of those datasets : cartography, urban planning, traffic analysis, biomass estimation and so on. Therefore, lots of progresses have been made to use machine learning to help us have a better understanding of our Earth Observation data.

In this work, we show that deep learning allows a computer to parse and classify objects in an image and can be used for automatical cartography from remote sensing data. Especially, we provide examples of deep fully convolutional networks that can be trained for semantic labeling for airborne pictures of urban areas.

## Content

### Deep networks

![](https://lut.im/pexiZxMS7n/MlVhwOQXHz1Va0Yl)

We provide a deep neural network based on the [SegNet architecture](https://arxiv.org/abs/1511.02680) for semantic labeling of Earth Observation images. The network is separated in two files :
  * The network definition (the architecture) in the prototxt format (for the [Caffe](https://github.com/bvlc/caffe) framework)
  * Pre-trained weights on the [ISPRS Vaihingen](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-vaihingen.html) dataset (IRRG only)

[segnet_vaihingen_128x128_fold1_iter_60000.caffemodel (112.4 Mo)](http://www.irisa.fr/obelix/files/audebert/segnet_vaihingen_128x128_fold1_iter_60000.caffemodel) ([backup link](https://drive.google.com/open?id=0B8XVGOkhuqDTTmh2UDFlYWdpV28)): pre-trained model on the ISPRS Vaihingen dataset (IRRG training tiles)

### Jupyter notebooks

In addition to our models, we provide several Jupyter notebooks to :
  * pre-process remote sensing images and build a dataset (with the example of the [ISPRS Vaihingen](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-vaihingen.html) dataset)
  * train a deep network for semantic labeling of remote sensing images
  * apply a deep network to label automatically remote sensing data

Configuration variables are grouped in the `config.py` file. This the one you **have to edit** to suit your needs. The scripts that can be used are :
  * `Extract images` : use a sliding window to divide high resolution images in smaller patches
  * `Create LMDB` : convert the small patches into LMDB for faster Caffe processing
  * `Training` : train a model on the LMDB

Those scripts are available either using Python or Jupyter. Two inference scripts are available but only for Python (though they can easily be imported in Jupyter).

## Requirements

You need to compile and use our Caffe fork (including [Alex Kendall's Unpooling Layer](https://github.com/alexgkendall/caffe-segnet)) to use the provided models. Training on GPU is recommended but not mandatory. You can download the fork by cloning this repository and executing :
```
# Clone this repository
git clone https://github.com/nshaud/DeepNetsForEO.git

cd DeepsNetsForEO/

# Initialize and checkout our custom Caffe fork (upsample branch !)
git submodule init

git submodule update
```

Our Caffe version will be available in `caffe` folder. You can then follow with the usual [compilation instructions](http://caffe.berkeleyvision.org/installation.html#compilation).

## References

If you use this work for your projects, please take the time to cite our ACCV'16 paper :

[https://arxiv.org/abs/1609.06846](https://arxiv.org/abs/1609.06846) Nicolas Audebert, Bertrand Le Saux and Sébastien Lefèvre, **Semantic Segmentation of Earth Observation Data Using Multimodal and Multi-scale Deep Networks**, *Asian Conference on Computer Vision*, 2016.
```
@inproceedings{audebert_semantic_2016,
    address = {Taipei, Taiwan},
    title = {Semantic {Segmentation} of {Earth} {Observation} {Data} {Using} {Multimodal} and {Multi}-scale {Deep} {Networks}},
    url = {https://hal.archives-ouvertes.fr/hal-01360166},
    urldate = {2016-10-13},
    booktitle = {Asian {Conference} on {Computer} {Vision} ({ACCV}16)},
    author = {Audebert, Nicolas and Le Saux, Bertrand and Lefèvre, Sébastien},
    month = nov,
    year = {2016},
    keywords = {computer vision, data fusion, Earth observation, Neural networks, remote sensing},
}
```

## Acknowledgements

This work has been conducted at ONERA ([DTIM](http://www.onera.fr/en/dtim)) and IRISA ([OBELIX team](https://www-obelix.irisa.fr/)), with the support of the joint Total-ONERA research project NAOMI.

The Vaihingen data set was provided by the German Society for Photogrammetry, Remote Sensing and Geoinformation (DGPF).
