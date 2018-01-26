# Deep learning for Earth Observation

![http://www.onera.fr/en/dtim](https://lut.im/qwL8UPM0en/soXh7XtiPp4IoML8.png)
![https://www-obelix.irisa.fr/](https://lut.im/D9OHdqk6rK/Mod6HUGCtwjY00Vk.png)
![](https://lut.im/5UpsUhIzgE/Fa6XCquMRsnHwjHu.png)

This repository contains code, network definitions and pre-trained models for working on remote sensing images using deep learning.

We build on the [SegNet architecture](https://github.com/alexgkendall/SegNet-Tutorial) (Badrinarayanan et al., 2015) to provide a semantic labeling network able to perform dense prediction on remote sensing data. The implementation uses the [PyTorch](https://pytorch.org) framework.

## Motivation

![](https://lut.im/YriLDf2Lb9/gaB9VlcBgZ6yy6l6.jpg)

Earth Observation consists in visualizing and understanding our planet thanks to airborne and satellite data. Thanks to the release of large amounts of both satellite (e.g. Sentinel and Landsat) and airborne images, Earth Observation entered into the Big Data era. Many applications could benefit from automatic analysis of those datasets : cartography, urban planning, traffic analysis, biomass estimation and so on. Therefore, lots of progresses have been made to use machine learning to help us have a better understanding of our Earth Observation data.

In this work, we show that deep learning allows a computer to parse and classify objects in an image and can be used for automatical cartography from remote sensing data. Especially, we provide examples of deep fully convolutional networks that can be trained for semantic labeling for airborne pictures of urban areas.

## Content

### Deep networks

![](https://lut.im/pexiZxMS7n/MlVhwOQXHz1Va0Yl)

We provide a deep neural network based on the [SegNet architecture](https://arxiv.org/abs/1511.02680) for semantic labeling of Earth Observation images.

All the pre-trained weights can be found on the [OBELIX team website](http://www-obelix.irisa.fr/software/) ([backup link](https://drive.google.com/open?id=1cwXe8ANkhFqe2i_UNxpZu15y2HZ0N9KN).

### Data

Our example models are trained on the [ISPRS Vaihingen dataset](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-vaihingen.html) and [ISPRS Potsdam dataset](http://www2.isprs.org/potsdam-2d-semantic-labeling.html). We use the IRRG tiles (8bit format) and we build 8bit composite images using the DSM, NDSM and NDVI. The ground truth files are color-encoded and should be converted to the numerical labels, e.g. {0,1,2,3,4,5} instead of {[255,255,255],[0,0,255],[0,255,255],[0,255,0],[255,0,255],[255,0,0]} using the `convert_gt.py` script.

You can either use our script from the `OSM` folder (based on the [Maperitive](http://maperitive.net/) software) to generate OpenStreetMap rasters from the images, or download the OSM tiles from Potsdam [here](https://drive.google.com/open?id=0B8XVGOkhuqDTdGNibWJPeTcxLVE).

### How to start

Just run the `SegNet_PyTorch_v2.ipynb` notebook using [Jupyter](https://jupyter.org/)!

## Requirements

Find the right version for your setup and install [PyTorch](https://pytorch.org).

Then, you can use `pip` or any package manager to install the packages listed in `requirements.txt`, e.g. by using:
```
pip install -r requirements.txt
```

## References

If you use this work for your projects, please take the time to cite our ISPRS Journal paper :

[https://arxiv.org/abs/1711.08681](https://arxiv.org/abs/1711.08681) Nicolas Audebert, Bertrand Le Saux and Sébastien Lefèvre, **Beyond RGB: Very High Resolution Urban Remote Sensing With Multimodal Deep Networks**, ISPRS Journal of Photogrammetry and Remote Sensing, 2017.

```
@article{audebert_beyond_2017,
title = "Beyond RGB: Very high resolution urban remote sensing with multimodal deep networks",
journal = "ISPRS Journal of Photogrammetry and Remote Sensing",
year = "2017",
issn = "0924-2716",
doi = "https://doi.org/10.1016/j.isprsjprs.2017.11.011",
author = "Nicolas Audebert and Bertrand Le Saux and Sébastien Lefèvre",
keywords = "Deep learning, Remote sensing, Semantic mapping, Data fusion"
}
```

## License

Code (scripts and Jupyter notebooks) are released under the GPLv3 license for non-commercial and research purposes **only**. For commercial purposes, please contact the authors.

![https://creativecommons.org/licenses/by-nc-sa/3.0/](https://i.creativecommons.org/l/by-nc-sa/3.0/nl/88x31.png) The network weights are released under Creative-Commons BY-NC-SA. For commercial purposes, please contact the authors.

See `LICENSE.md` for more details.

## Acknowledgements

This work has been conducted at ONERA ([DTIM](http://www.onera.fr/en/dtim)) and IRISA ([OBELIX team](https://www-obelix.irisa.fr/)), with the support of the joint Total-ONERA research project NAOMI.

The Vaihingen data set was provided by the German Society for Photogrammetry, Remote Sensing and Geoinformation (DGPF).
