from __future__ import print_function
import os
import argparse
from osgeo import gdal, osr
import warnings
from skimage import io
from skimage.transform import resize
from skimage.util import crop
import numpy as np
from copy_geoinfo import copy_geoinfo
from tqdm import tqdm
from labels import convert_to_color, convert_from_color
import subprocess


# From : http://gis.stackexchange.com/questions/57834/how-to-get-raster-corner-coordinates-using-python-gdal-bindings
def GetExtent(gt, cols, rows):
    ''' Return list of corner coordinates from a geotransform

        @type gt:   C{tuple/list}
        @param gt: geotransform
        @type cols:   C{int}
        @param cols: number of columns in the dataset
        @type rows:   C{int}
        @param rows: number of rows in the dataset
        @rtype:    C{[float,...,float]}
        @return:   coordinates of each corner
    '''
    ext = []
    xarr = [0, cols]
    yarr = [0, rows]

    for px in xarr:
        for py in yarr:
            x = gt[0] + (px * gt[1]) + (py * gt[2])
            y = gt[3] + (px * gt[4]) + (py * gt[5])
            ext.append([x, y])
        yarr.reverse()
    return ext


def ReprojectCoords(coords, src_srs, tgt_srs):
    ''' Reproject a list of x,y coordinates.

        @type geom:     C{tuple/list}
        @param geom:    List of [[x,y],...[x,y]] coordinates
        @type src_srs:  C{osr.SpatialReference}
        @param src_srs: OSR SpatialReference object
        @type tgt_srs:  C{osr.SpatialReference}
        @param tgt_srs: OSR SpatialReference object
        @rtype:         C{tuple/list}
        @return:        List of transformed [[x,y],...[x,y]] coordinates
    '''
    trans_coords = []
    transform = osr.CoordinateTransformation(src_srs, tgt_srs)
    for x, y in coords:
        x, y, z = transform.TransformPoint(x, y)
        trans_coords.append([x, y])
    return trans_coords


def getCorners(raster):
        ds = gdal.Open(raster)
        gt = ds.GetGeoTransform()
        cols = ds.RasterXSize
        rows = ds.RasterYSize
        ext = GetExtent(gt, cols, rows)

        src_srs = osr.SpatialReference()
        src_srs.ImportFromWkt(ds.GetProjection())
        tgt_srs = osr.SpatialReference()
        tgt_srs.ImportFromEPSG(reference_srs_numeric)
        tgt_srs = src_srs.CloneGeogCS()

        geo_ext = ReprojectCoords(ext, src_srs, tgt_srs)
        return geo_ext

parser = argparse.ArgumentParser()
parser.add_argument("images", help="Base images to process (at least one)",
                    nargs="+")
parser.add_argument("--source", help="Source OSM file (.osm)",
                    type=str)
parser.add_argument("--out", help="Folder where to save the OSM tiles",
                    type=str)
parser.add_argument("--scale", help="Scale for Maperitive. "
                    "Increase it the OSM temporary maps are too small, or "
                    "decrease it if they are too big (default : 4).",
                    type=int, default=4)
parser.add_argument("--rules", help="Rulefile for Maperitive.",
                    type=str, default="./Custom.mrules")
parser.add_argument("--maperitive", help="Maperitive binary.",
                    type=str, default="./Maperitive.exe")
args = parser.parse_args()

files = args.images
dest = args.out
if not os.path.isdir(dest):
    print("Creating output folder {}".format(dest))
    try:
        os.mkdir(dest)
    except Exception as e:
        print("Unable to create destination folder. Abort")
        raise e
source = args.source
SCALE = args.scale
maperitive_binary = os.path.abspath(args.maperitive)
RULEFILE = os.path.abspath(args.rules)

print("")
print("###############################################")
print("##### Georeferenced images to OSM rasters #####")
print("###############################################")
print("")

# Find the projection of the dataset
f = files[0]
ds = gdal.Open(f)
prj = ds.GetProjection()
srs = osr.SpatialReference(wkt=prj)
gdal_srs_value = ":".join([srs.GetAttrValue('AUTHORITY', 0),
                           srs.GetAttrValue('AUTHORITY', 1)])
print("The dataset uses the {srs} projection system.".
      format(srs=gdal_srs_value))
# Maperitive (and OpenStreetMap) use EPSG:3857 as
# default rendering projection
reference_srs_numeric = 3857
reference_srs_value = 'EPSG:3857'

projected_files = []
print("Projecting input files from {in_srs} to {ref_srs}".
      format(in_srs=gdal_srs_value, ref_srs=reference_srs_value))
tfiles = tqdm(files)
for f in tfiles:
    filename = f.split('/')[-1].split('.')[0]
    tfiles.set_description("Projection of {} OK".format(filename))
    projected_f = dest + filename + '_reproj.tif'
    projected_files.append(projected_f)
    # Create the projection only if the file does not exist
    # WARNING : this might do some weird stuff if the file exists
    #           but in the wrong projection.
    if not os.path.isfile(projected_f):
        command = 'gdalwarp -overwrite -s_srs {in_srs} '
        command += '-t_srs {target_srs} -of GTiff {file_in} {file_out}'
        command = command.format(file_in=f, file_out=projected_f,
                                 in_srs=gdal_srs_value,
                                 target_srs=reference_srs_value)
        try:
            subprocess.check_output(command, shell=True)
        except Exception as e:
            message = "GDAL failed to process {file_in}.\n".format(file_in=f)
            message += "Please check that GDAL is installed and that the "\
                       "requested files are readable."
            print("Traceback : ", e)
            raise Exception(message)

coords_list = [getCorners(f) for f in projected_files]

ms = []
ms.append('set-setting name=map.decoration.grid value=False')
ms.append('use-ruleset location={}'.format(RULEFILE))
ms.append('load-source {}'.format(source))
ms.append('set-setting name=map.decoration.attribution value=false')
ms.append('set-setting name=map.decoration.scale value=false')

triples = []
for coords, p_f, f in zip(coords_list, projected_files, files):
        filename = f.split('/')[-1].split('.')[0]
        lngs, lats = zip(*coords)
        minlng, minlat = min(lngs), min(lats)
        maxlng, maxlat = max(lngs), max(lats)
        ms.append('set-geo-bounds {}, {}, {}, {}'.
                  format(minlng, minlat, maxlng, maxlat))
        ms.append('set-print-bounds-geo')
        ms.append('zoom-map-scale 10')
        dest_filename = '{}{}_osm.png'.format(dest, filename)
        ms.append('export-bitmap world-file=true scale={scale} file={dest}'.
                  format(scale=SCALE, dest=dest_filename))
        triples.append((f, p_f, dest_filename))

# 2. run the Maperitive script
temp_script_file_name = '/tmp/maperitive_script.txt'
print('Running Maperitive script: {}'.format(temp_script_file_name))

with open(temp_script_file_name, 'w') as maperitive_script_file:
    for a_line in ms:
        # print(a_line)
        maperitive_script_file.write(a_line + '\n')

maperitive_script_file.close()

# Call Maperitive and hide the output by redirecting to /dev/null
subprocess.call(maperitive_binary + ' -exitafter ' + temp_script_file_name,
                shell=True,
                stdout=open(os.devnull, 'wb'),
                stderr=open(os.devnull, 'wb'))

print("Projecting back the OSM rasters into {}.".format(gdal_srs_value))

for rgb_f, projected_f, osm_f in tqdm(triples):
    projected_rgb = io.imread(projected_f)
    osm_raster = io.imread(osm_f)

    # Remove the alpha channel from the Maperitive rendering
    osm_raster = osm_raster[:, :, :3]
    # Resize to the projection tile size
    # (Nearest neighbours, no interpolation)
    osm_raster = resize(osm_raster, projected_rgb.shape[:2],
                        order=0, preserve_range=True)
    osm_raster = np.asarray(osm_raster, dtype='uint8')

    projected_osm_f = osm_f + '_reproj.tif'

    # Keep only the OSM values that are in the original image
    rgb = np.sum(projected_rgb, axis=-1)
    rgb_mask = rgb == 0
    osm_raster[rgb_mask] = [0, 0, 0]

    # Save the projected (in ESPG:3857) OSM raster
    io.imsave(projected_osm_f, osm_raster)

    # Copy the georeference from projected RGB to OSM raster
    copy_geoinfo(projected_f, projected_osm_f)

    matching_osm_f = projected_osm_f + '_matching.tif'
    # Reproject the OSM raster into original georeference
    command = 'gdalwarp -overwrite -s_srs {in_srs} -t_srs {out_srs} '\
              '-of GTiff {osm_projected} {osm_matched}'
    try:
        subprocess.check_output(command.format(osm_projected=projected_osm_f,
                                               osm_matched=matching_osm_f,
                                               in_srs=reference_srs_value,
                                               out_srs=gdal_srs_value),
                                               shell=True)
    except Exception as e:
        message = "GDAL failed to process {file_in}.\n".format(file_in=f)
        message += "Please check that GDAL is installed and that the "\
                   "requested files are readable."
        print("Traceback : ", e)
        raise Exception(message)

    # Delete the temporary file
    os.rename(matching_osm_f, projected_osm_f)

    img = io.imread(projected_osm_f)
    w, h = img.shape[:2]
    ref_w, ref_h = io.imread(rgb_f).shape[:2]
    offset_w, offset_h = w - ref_w, h - ref_h
    if offset_w % 2 == 0:
        crop_w = (offset_w // 2, offset_w // 2)
    else:
        crop_w = (offset_w // 2, offset_w // 2 + 1)
    if offset_h % 2 == 0:
        crop_h = (offset_h // 2, offset_h // 2)
    else:
        crop_h = (offset_h // 2, offset_h // 2 + 1)

    # Crop the OSM image to the target tile size
    img = crop(img, (crop_w, crop_h, (0, 0)))

    labels_osm = convert_from_color(img)

    color_osm = convert_to_color(labels_osm)

    final_osm_file = osm_f.replace('.png', '_{}.tif')
    # Ignore low contrast warning from scikit-image
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        io.imsave(final_osm_file.format('labels'), labels_osm)
    final_osm_file = osm_f.replace('.png', '_{}.png')
    io.imsave(final_osm_file.format('colors'), color_osm)

    # Cleaning
    os.remove(osm_f)
    os.remove(osm_f.replace('png', 'pgw'))
    os.remove(osm_f + '.georef')
    os.remove(projected_osm_f)

print("OSM files have been saved in {}".format(dest))
