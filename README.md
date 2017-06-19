# Deep learning for Earth Observation

![http://www.onera.fr/en/dtim](https://lut.im/hCESNeToVY/qrK3KIIi1gcF8ZJD)
![https://www-obelix.irisa.fr/](https://lut.im/gaJPw4ny09/7YziLBwPR2o0kdKd)
![](https://lut.im/i5zoaeshn2/w7IKoiAfoNZGDVmq)

This repository contains code, network definitions and pre-trained models for working on remote sensing images using deep learning.

We build on the [SegNet architecture](https://github.com/alexgkendall/SegNet-Tutorial) (Badrinarayanan et al., 2015) to provide a semantic labeling network able to perform dense prediction on remote sensing data.

## Motivation

![](https://lut.im/YriLDf2Lb9/gaB9VlcBgZ6yy6l6.jpg)

Earth Observation consists in visualizing and understanding our planet thanks to airborne and satellite data. Thanks to the release of large amounts of both satellite (e.g. Sentinel and Landsat) and airborne images, Earth Observation entered into the Big Data era. Many applications could benefit from automatic analysis of those datasets : cartography, urban planning, traffic analysis, biomass estimation and so on. Therefore, lots of progresses have been made to use machine learning to help us have a better understanding of our Earth Observation data.

In this work, we show that deep learning allows a computer to parse and classify objects in an image and can be used for automatical cartography from remote sensing data. Especially, we provide examples of deep fully convolutional networks that can be trained for semantic labeling for airborne pictures of urban areas.

## Content

### Deep networks

![](https://lut.im/pexiZxMS7n/MlVhwOQXHz1Va0Yl)

We provide a deep neural network based on the [SegNet architecture](https://arxiv.org/abs/1511.02680) for semantic labeling of Earth Observation images. The network is separated in two files :
  * The network definition (the architecture) in the prototxt format (for the [Caffe](https://github.com/bvlc/caffe) framework),
  * Pre-trained weights on the [ISPRS Vaihingen](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-vaihingen.html) dataset and [ISPRS Potsdam](http://www2.isprs.org/potsdam-2d-semantic-labeling.html) datasets.

All the pre-trained weights can be found on the [OBELIX team website](http://www-obelix.irisa.fr/software/) ([backup link](https://drive.google.com/open?id=0B8XVGOkhuqDTaUE0OUJNQ21kOWc)).

Examples :
  - [segnet_vaihingen_128x128_fold1_iter_60000.caffemodel (112.4 Mo)](http://www.irisa.fr/obelix/files/audebert/segnet_vaihingen_128x128_fold1_iter_60000.caffemodel) ([backup link](https://drive.google.com/open?id=0B8XVGOkhuqDTTmh2UDFlYWdpV28)): pre-trained model on the ISPRS Vaihingen dataset (trained on tiles 1, 3, 5, 7, 11, 13, 15, 17, 21, 23, 26, 28, 30, validated on tiles 32, 34, 37).
  - [potsdam_rgb_128_fold1_iter_80000.caffemodel (112.4 Mo)]() ([backup link](https://drive.google.com/open?id=0B8XVGOkhuqDTT0lCbVBDVEtCTXM)) : pre-trained model on the ISPRS Potsdam dataset (RGB tiles, trained on (3, 12), (6, 8), (4, 11), (3, 10), (7, 9), (4, 10), (6, 10), (7, 7), (5, 10), (7, 11), (2, 12), (6, 9), (5, 11), (6, 12), (7, 8), (2, 10), (6, 7), (6, 11), validated on tile (2, 11), (7, 12), (3, 11), (5, 12), (7, 10), (4, 12)).

### Data

Our example models are trained on the [ISPRS Vaihingen dataset](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-vaihingen.html) and [ISPRS Potsdam dataset](). We use the IRRG tiles (8bit format) and we build 8bit composite images using the DSM, NDSM and NDVI. The ground truth files are color-encoded and should be converted to the numerical labels, e.g. {0,1,2,3,4,5} instead of {[255,255,255],[0,0,255],[0,255,255],[0,255,0],[255,0,255],[255,0,0]} using the `convert_gt.py` script.

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

### How to start

This an example of how to start using the ISPRS Vaihingen dataset. This dataset contains orthoimages (`top/` folder) and ground truthes (`gts_for_participants/`) folder.

  1. First, we need to edit the `config.py` file. `BASE_DIR`, `DATASET` and `DATASET_DIR` are used to point to the folder where the dataset is stored and to specify a unique name for the dataset, e.g. "Vaihingen". `label_values` and `palette` define the classes that will be used and the associated colors in RGB format. `folders`, `train_ids` and `test_ids` define the folder arrangement of the dataset and the train/test split using unique numerical ids associated to the tiles.
  2. We need to transform the ground truth RGB-encoded images to 2D matrices. We can use the `convert_gt.py` script to do so, e.g. : `python convert_gt.py gts_for_participants/*.tif --from-color --out gts_numpy/`. This will populate a new `gts_numpy/` folder containing the matrices. Please note that the `folders` value for the labels should point to this folder (`gts_numpy/`).
  3. Extract small patches from the tiles to create the train and test sets : `python extract_images.py`
  4. Populate LMDB using the extracted images : `python create_lmdb.py`
  5. Train the network for 40 000 iterations, starting with VGG-16 weights and save the weights into the `trained_network_weights` folder : `python training.py --niter 40000 --update 1000 --init vgg16weights.caffemodel --snapshot trained_network_weights/`
  6. Test the trained network on some tiles : `python inference.py 16 32 --weights trained_network_weights/net_iter_40000.caffemodel`

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
	title = {Semantic {Segmentation} of {Earth} {Observation} {Data} {Using} {Multimodal} and {Multi}-scale {Deep} {Networks}},
	url = {https://link.springer.com/chapter/10.1007/978-3-319-54181-5_12},
	doi = {10.1007/978-3-319-54181-5_12},
	language = {en},
	urldate = {2017-03-31},
	booktitle = {Computer {Vision} – {ACCV} 2016},
	publisher = {Springer, Cham},
	author = {Audebert, Nicolas and Le Saux, Bertrand and Lefèvre, Sébastien},
	month = nov,
	year = {2016},
	pages = {180--196}
}
```
## License

Code (scripts and Jupyter notebooks) are released under the GPLv3 license for non-commercial and research purposes **only**. For commercial purposes, please contact the authors.

![https://creativecommons.org/licenses/by-nc-sa/3.0/](https://i.creativecommons.org/l/by-nc-sa/3.0/nl/88x31.png) The network weights are released under Creative-Commons BY-NC-SA. For commercial purposes, please contact the authors.

See `LICENSE.md` for more details.

## Acknowledgements

This work has been conducted at ONERA ([DTIM](http://www.onera.fr/en/dtim)) and IRISA ([OBELIX team](https://www-obelix.irisa.fr/)), with the support of the joint Total-ONERA research project NAOMI.

The Vaihingen data set was provided by the German Society for Photogrammetry, Remote Sensing and Geoinformation (DGPF).
