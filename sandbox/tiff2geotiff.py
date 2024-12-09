import argparse
import os
import sys
import time
import yaml

import cv2
import glob
import numpy as np 
from osgeo import gdal

sys.path.insert(1, '.')
import functions.fct_misc as misc
from functions.constants import DONE_MSG

from loguru import logger
logger = misc.format_logger(logger)

cwd = os.getcwd()

if __name__ == "__main__":

    # Start chronometer
    tic = time.time()
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('config_file', type=str, help='Framework configuration file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")
 
    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    GEOTIFF_IMAGE_DIR = cfg['geotiff_image_dir']
    TIFF_IMAGE_DIR = cfg['tiff_image_dir']

    logger.info('----- read data1 -------')

    filesTs_list =  glob.glob(os.path.join(cwd + GEOTIFF_IMAGE_DIR, '*.tif'))
    files_name1 = [fn.split('\\')[-1].split('.tif')[0].strip() for fn in filesTs_list]
    N = len(filesTs_list)

    for j1 in range(N):
        logger.info(f"final2 image = {j1}")
        logger.info(files_name1[j1][:])                
                    
        ima = cv2.imread(filesTs_list[j1][:])
        ima0 = ima

        dataset1 = gdal.Open(filesTs_list[j1][:])
        projection = dataset1.GetProjection()
        geotransform = dataset1.GetGeoTransform()

        file_name = files_name1[j1].split("/")
        ima=cv2.imread(cwd + GEOTIFF_IMAGE_DIR + file_name[-1]+'.tif')
        dataset2 = gdal.Open(cwd + TIFF_IMAGE_DIR + file_name[-1]+'.tif', gdal.GA_Update)
        dataset2.SetGeoTransform(geotransform)
        dataset2.SetProjection( projection )

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")
    
    sys.stderr.flush()