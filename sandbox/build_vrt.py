import os
from time import time

from osgeo import gdal

tic = time()

WORKING_DIRECTORY = 'output/det/vaud/tiles_z18/'
os.chdir(WORKING_DIRECTORY)

print('Create VRT file for each year')
for year in range(1980, 1981):
    SOURCE_FOLDER = str(year)
    if not os.path.exists(SOURCE_FOLDER):
        continue

    VRT_NAME =  os.path.join(SOURCE_FOLDER, SOURCE_FOLDER + '.vrt')
    EPSG = 'EPSG:3857'

    list_im = []

    for file in os.listdir(SOURCE_FOLDER):
        
        if not file.endswith(("tif", "tiff")):
            continue
        
        list_im.append(os.path.join(SOURCE_FOLDER, file))

    if len(list_im) == 0:
        print(f"The folder {SOURCE_FOLDER} does not contain any image.")
        continue

    vrt_options = gdal.BuildVRTOptions(resampleAlg='cubic', addAlpha=False, outputSRS=EPSG)
    my_vrt = gdal.BuildVRT(VRT_NAME, list_im, options=vrt_options)
    my_vrt = None

    print(f"The file {VRT_NAME} has been created.")

toc = time()
print(f"Ellapsed time: {toc-tic}.")