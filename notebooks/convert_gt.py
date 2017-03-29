from __future__ import print_function
import numpy as np
from skimage import io
from tqdm import tqdm
import argparse
import os
from config import palette, invert_palette


def convert_to_color(arr_2d, palette=palette):
    """ grayscale labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d


def convert_from_color(arr_3d, palette=invert_palette):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("images", help="Images to process (at least one)",
                        nargs='+')
    parser.add_argument("--to-color",
                        help="Convert from grayscale labels"
                             "to RGB encoded labels",
                        action="store_true")
    parser.add_argument("--from-color",
                        help="Convert from RGB encoded labels"
                             "to grayscale labels",
                        action="store_true")
    parser.add_argument("--out",
                        help="Folder where to save the modified images",
                        type=str)
    args = parser.parse_args()

    files = args.images

    if args.to_color and args.from_color:
        raise ValueError("Cannot specify both --from-color"
                         "and --to-color at the same time")
    elif args.to_color:
        convert_fun = convert_to_color
    elif args.from_color:
        convert_fun = convert_from_color
    else:
        raise ValueError("You need to specify whether to convert"
                         "from or to the RGB color labels")

    if args.out is None:
        OUTPUT_FOLDER = './out'
    else:
        OUTPUT_FOLDER = args.out

    if os.path.isdir(OUTPUT_FOLDER):
        print("WARNING : output folder {} exists !".format(OUTPUT_FOLDER))
    else:
        os.mkdir(OUTPUT_FOLDER)

    for f in tqdm(files):
        filename = f.split('/')[-1]
        img = io.imread(f)
        new_img = convert_fun(img)
        io.imsave(OUTPUT_FOLDER + '/' + filename, new_img)
