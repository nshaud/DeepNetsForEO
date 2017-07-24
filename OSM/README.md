# Script to download OSM rasters from georeferenced images

Requirements :
  * install the Python packages from requirements.txt (basically, numpy, GDAL, scikit-image)
  * [Maperitive](http://maperitive.net/)
  * the OSM vector information from the zone you're interested in. You can use QGIS to download this.

The script takes as input a georeferenced image. It then reprojects this image into EPSG:3857 and grab the corners. It then calls Maperitive to render the corresponding raster, using our custom rules (basically, our rules assign a specific color for each OSM category of interest). Then, the color raster is converted into a label map.
