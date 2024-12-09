import os
import sys
import time
import argparse
import yaml
from loguru import logger
from rasterio.plot import show
from rasterio.merge import merge
import rasterio as rio
from pathlib import Path

sys.path.insert(1, '.')
import functions.fct_misc as misc

logger = misc.format_logger(logger)


if __name__ == "__main__":

    # Start chronometer
    tic = time.time()
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description="Mosaic image")
    parser.add_argument('config_file', type=str, help='Framework configuration file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")
 
    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    WORKING_DIR = cfg['working_dir']
    INPUT_DIR = cfg['input_dir']
    OUTPUT_DIR = cfg['output_dir']

    os.chdir(WORKING_DIR)

    # Create an output directory in case it doesn't exist
    misc.ensure_dir_exists(os.path.join(OUTPUT_DIR))

    # Get raster file names
    raster_files = list(Path(INPUT_DIR).iterdir())
    raster_to_mosaic = []

    # Get filename
    split = INPUT_DIR.split('/')
    zoom = split[-4].split('_')[1]
    year = split[-3]
    filename = zoom + '_' + year + '_mosaic.tif'

    # Mosaic images and save
    for p in raster_files:
        raster = rio.open(p)
        raster_to_mosaic.append(raster)

    mosaic, output = merge(raster_to_mosaic)

    output_meta = raster.meta.copy()
    output_meta.update(
        {"driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": output,
        })

    with rio.open(os.path.join(OUTPUT_DIR, filename), 'w', **output_meta) as m:
        m.write(mosaic)