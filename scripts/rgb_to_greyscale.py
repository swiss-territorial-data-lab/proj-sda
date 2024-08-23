import argparse
import os
import sys
import time

import numpy as np
import rasterio as rio

from glob import glob
from loguru import logger
from rasterio.plot import show
from tqdm import tqdm
from yaml import load, FullLoader

sys.path.insert(1, '.')
import functions.fct_misc as misc
from functions.constants import DONE_MSG

logger = misc.format_logger(logger)


def main(WORKING_DIR, OUTPUT_DIR, IMAGE_DIR, BANDS):

    os.chdir(WORKING_DIR)

    # Create output directories in case they don't exist
    _ = misc.ensure_dir_exists(OUTPUT_DIR)

    images = glob(os.path.join(IMAGE_DIR, '*.tif'))
    for image in tqdm(images, desc='Convert RGB images to greyscale images', total=len(images)):

        with rio.open(image) as src:
            r = src.read()
            R = src.read(1)
            G = src.read(2)
            B = src.read(3)
            # https://en.wikipedia.org/wiki/Grayscale#Colorimetric_(perceptual_luminance-preserving)_conversion_to_grayscale 
            greyscale = 0.2990*R + 0.5870*G + 0.1140*B
            # rio.plot.show(greyscale, cmap='Greys_r')

        img_name = os.path.basename(image)

        if BANDS == 1:
            output_meta = src.meta.copy()
            output_meta.update(
                {"driver": "GTiff",
                    "height": r.shape[1],
                    "width": r.shape[2],
                    "count": 1,
                    "crs": src.crs
                })
            
            with rio.open(os.path.join(OUTPUT_DIR, img_name), 'w', **output_meta) as dst:
                dst.write(greyscale, 1)

        elif BANDS == 3:
            grey_image_arr = np.expand_dims(greyscale, -1)
            grey_image_arr_3_channel = grey_image_arr.repeat(3, axis=-1)

            output_meta = src.meta.copy()
            output_meta.update(
                {"driver": "GTiff",
                    "height": r.shape[1],
                    "width": r.shape[2],
                    "count": 3,
                    "crs": src.crs
                })

            with rio.open(os.path.join(OUTPUT_DIR, img_name), 'w', **output_meta) as dst:
                dst.write(np.moveaxis(grey_image_arr_3_channel, [0, 1, 2], [2, 1, 0]))


if __name__ == "__main__":

    # Start chronometer
    tic = time.time()
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description="Convert RGB images to grayscale (STDL.proj-sda)")
    parser.add_argument('config_file', type=str, help='Framework configuration file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")
 
    with open(args.config_file) as fp:
        cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    WORKING_DIR = cfg['working_dir']
    IMAGE_DIR = cfg['image_dir']
    OUTPUT_DIR = cfg['output_dir']
    BANDS = cfg['bands']

    main(WORKING_DIR, OUTPUT_DIR, IMAGE_DIR, BANDS)

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()