######################## DATASET PARAMETERS ###############################

"""
    These values are used for training (when building the dataset)
    patch_size : (x,y) size of the patches to be extracted (usually square)
    step_size : stride of the sliding. If step_size < min(patch_size), then
                there will be an overlap.
"""
patch_size = (128, 128)
step_size = 32

""" ROTATIONS :
    For square patches, valid rotations are 90, 180 and 270.
    e.g. : [] for no rotation, [180] for only 180 rotation, [90, 180]...
"""
ROTATIONS = []
""" FLIPS :
    [False, False] : no symetry
    [True, False] : up/down symetry only
    [False, True] : left/right symetry only
    [True, True] : both up/down and left/right symetries
"""
FLIPS = [False, False]

"""
    BASE_DIR: main dataset folder
    DATASET : dataset name (using for later naming)
    DATASET_DIR : where the current dataset is stored
    FOLDER_SUFFIX : suffix to distinguish this dataset from others (optional)
    BASE_FOLDER : the base folder for the dataset
    BGR : True if we want to reverse the RGB order (Caffe/OpenCV convention)
    label_values : string names for the classes
"""
BASE_DIR = 'ISPRS/'
DATASET = 'Vaihingen'
FOLDER_SUFFIX = '_fold1'
BASE_FOLDER = BASE_DIR + DATASET + '/'
BGR = True
label_values = ['imp_surfaces', 'building', 'low_vegetation',
                'tree', 'car', 'clutter']
# Color palette
palette = {0: (255, 255, 255),  # Impervious surfaces (white)
           1: (0, 0, 255),      # Buildings (dark blue)
           2: (0, 255, 255),    # Low vegetation (light blue)
           3: (0, 255, 0),      # Tree (green)
           4: (255, 255, 0),    # Car (yellow)
           5: (255, 0, 0),      # Clutter (red)
           6: (0, 0, 0)}        # Unclassified (black)
invert_palette = {(255, 255, 255): 0,  # Impervious surfaces (white)
                  (0, 0, 255): 1,      # Buildings (dark blue)
                  (0, 255, 255): 2,    # Low vegetation (light blue)
                  (0, 255, 0): 3,      # Tree (green)
                  (255, 255, 0): 4,    # Car (yellow)
                  (255, 0, 0): 5,      # Clutter (red)
                  (0, 0, 0): 6}        # Unclassified (black)
NUMBER_OF_CLASSES = len(label_values)

"""
    The folders sequence lists the collections of the dataset, e.g. :
    BaseDirectory/
        MyDataset/
            aux_data/
            data/
            ground_truth/
    Each tuple inside the sequence has :
        (the collection name,
         the subfolder where the collection is stored,
         the filename format)
    For example, if we have : aux_data/aux_1.jpg, aux_data/aux_2.jpg, ...
                              data/1_data.jpg, data/1_data.jpg, ...
                              ground_truth/1_gt.jpg, ground_truth/2_gt.jpg, ...
    The folders variable should look like :
    folders = [
        ('aux_data', BASE_FOLDER + 'aux_data/', 'aux_{}.jpg',
         'data', BASE_FOLDER + 'data/', '{}_data.jpg',
         'ground_truth', BASE_FOLDER + 'ground_truth', '{}_gt.jpg')
    ]
    train_ids and test_ids should detail how to fill the {} in the name format.

    See the examples for the ISPRS Vaihingen and Potsdam for more details.
"""
if DATASET == 'Potsdam':
    folders = [
        ('labels', BASE_FOLDER + 'gts_numpy/', 'top_potsdam_{}_{}_label.png'),
        ('rgb', BASE_FOLDER + '2_Ortho_RGB/', 'top_potsdam_{}_{}_RGB.tif'),
        ('irrg', BASE_FOLDER + 'Y_Ortho_IRRG/', 'top_potsdam_{}_{}_IRRG.tif'),
        ('irgb', BASE_FOLDER + 'X_Ortho_IRGB/', 'top_potsdam_{}_{}_IRGB.tif')
    ]
    train_ids = [
         (3, 12), (6, 8), (4, 11), (3, 10), (7, 9), (4, 10), (6, 10), (7, 7),
         (5, 10), (7, 11), (2, 12), (6, 9), (5, 11), (6, 12), (7, 8), (2, 10),
         (6, 7), (6, 11), (4, 12)]
    test_ids = [(2, 11), (7, 12), (3, 11), (5, 12), (7, 10)]

elif DATASET == 'Vaihingen':
    folders = [
        ('labels', BASE_FOLDER + 'gts_numpy/', 'top_mosaic_09cm_area{}.png'),
        ('irrg', BASE_FOLDER + 'top/', 'top_mosaic_09cm_area{}.tif')
    ]
    train_ids = [(1,), (3,), (5,), (7,), (11,), (13,), (15,),
                 (17,),(21,), (23,), (26,), (28,), (30,)]
    test_ids = [(32,), (34,), (37,)]

# Build the target folder name
DATASET_DIR = BASE_FOLDER + DATASET.lower() + '_{}_{}_{}'.format(
                                    patch_size[0], patch_size[1], step_size)
# Add the suffix is not empty
if FOLDER_SUFFIX:
    DATASET_DIR += FOLDER_SUFFIX

DATASET_DIR += '/'

######################## LMDB PARAMETERS ###############################

""" LMDB to create in a format (source_folder, target_folder) """
data_lmdbs = [(DATASET_DIR + 'irrg_train/', DATASET_DIR + 'irrg_train_lmdb')]

test_lmdbs = [(DATASET_DIR + 'irrg_test/', DATASET_DIR + 'irrg_test_lmdb')]

label_lmdbs = [(DATASET_DIR + 'labels_train/', DATASET_DIR + 'labels_train_lmdb')]

test_label_lmdbs = [(DATASET_DIR + 'labels_test/', DATASET_DIR + 'labels_test_lmdb')]

######################## TESTING PARAMETERS ###############################

"""
    These values are used for testing (when evaluating new data)
    test_patch_size : (x,y) size of the patches to be extracted (usually square)
    test_step_size : stride of the sliding. If step_size < min(patch_size),
                there will be an overlap.
"""
test_patch_size = (128, 128)
test_step_size = 64

######################## CAFFE PARAMETERS ###############################

"""
    CAFFE_ROOT = path to Caffe local installation
    MODEL_FOLDER = where to store the model files
    INIT_MODEL = path to initialization weights (.caffemodel) or None
    CAFFE_MODE = 'gpu' or 'cpu'
    CAFFE_DEVICE = id of the gpu to use (if CAFFE_MODE is 'gpu')
    IGNORE_LABEL = label to ignore when classifying (e.g. clutter)
    TRAIN/TEST_DATA/LABEL_SOURCE = the LMDB containing train/test data and labels
    MEAN_PIXEL = the mean pixel to remove as data normalization (or None)
    BATCH_SIZE = batch size of the network (adjust according to available memory)
"""
CAFFE_ROOT = './caffesegnet/'
SOLVER_FILE = './reference_models/solver.prototxt'
MODEL_FOLDER = './'
INIT_MODEL = './reference_models/vgg16fc.caffemodel'
CAFFE_MODE = 'gpu'
CAFFE_DEVICE = 0
IGNORE_LABEL = 5
TRAIN_DATA_SOURCE = DATASET_DIR + 'irrg_train_lmdb'
TRAIN_LABEL_SOURCE = DATASET_DIR + 'labels_train_lmdb'
TEST_DATA_SOURCE = DATASET_DIR + 'irrg_test_lmdb'
TEST_LABEL_SOURCE = DATASET_DIR + 'labels_test_lmdb'
MEAN_PIXEL = [81.29, 81.93, 120.90]
BATCH_SIZE=10
