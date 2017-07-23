import numpy as np

# Color palette
palette = {
           0: (0, 0, 0),        # Undefined (black)
           1: (255, 255, 255),  # Impervious surfaces (white)
           2: (0, 0, 255),      # Buildings (dark blue)
           3: (0, 128, 0),      # Vegetation (light green)
           4: (255, 0, 0),      # Water (red)
          }

invert_palette = {v: k for k, v in palette.items()}

def convert_from_color(arr_3d, palette=invert_palette):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d

def convert_to_color(arr_2d, palette=palette):
    """ grayscale labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d
