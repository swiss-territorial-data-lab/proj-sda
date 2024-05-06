import argparse
import os
import sys
import time

from glob import glob
from loguru import logger
from PIL import Image
from tqdm import tqdm
from yaml import load, FullLoader

# the following allows us to import modules from within this file's parent folder
sys.path.insert(1, '.')
import functions.fct_misc as misc

logger = misc.format_logger(logger)


def main(WORKING_DIR, OUTPUT_DIR, IMAGE_DIR):

    os.chdir(WORKING_DIR)

    # Create output directories in case they don't exist
    _ = misc.ensure_dir_exists(OUTPUT_DIR)

    written_files = []

    images = glob(os.path.join(IMAGE_DIR, '*.tif'))

    for image in tqdm(images, desc='Convert RGB images to grayscale images', total=len(images)):
        img = Image.open(image).convert('L')
        img_name = os.path.basename(image)
        img.save(os.path.join(OUTPUT_DIR, img_name))

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

    main(WORKING_DIR, OUTPUT_DIR, IMAGE_DIR)

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()