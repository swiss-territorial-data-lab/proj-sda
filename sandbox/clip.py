import os
import sys
import time
import argparse
import yaml

import geopandas as gpd
from glob import glob
from loguru import logger
from tqdm import tqdm

# the following allows us to import modules from within this file's parent folder
sys.path.insert(1, '.')
import functions.misc as misc

logger = misc.format_logger(logger)


if __name__ == "__main__":

    # Start chronometer
    tic = time.time()
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description="The script clip MES detections to the AoI of the sda project (STDL.proj-sda)")
    parser.add_argument('config_file', type=str, help='Framework configuration file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")
 
    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    WORKING_DIR = cfg['working_dir']
    AOI_SHP = cfg['aoi_shapefile']
    DETECTIONS_DIR = cfg['detections_shapefile']
    OUTPUT_DIR = cfg['output_dir']

    os.chdir(WORKING_DIR)

    # Create an output directory in case it doesn't exist
    misc.ensure_dir_exists(os.path.join(OUTPUT_DIR))

    written_files = []

    logger.info(f"Read the AoI shapefile")
    aoi_gdf = gpd.read_file(AOI_SHP)

    logger.info(f"Read the detections shapefiles")
    detections = glob(os.path.join(DETECTIONS_DIR, '*.geojson'))

    feature_path = os.path.join(OUTPUT_DIR, 'MES_detections.gpkg')

    for detection in tqdm(detections, desc='Clip detections with AoI', total=len(detections)):

        detection_gdf = gpd.read_file(detection)
        detection_clip = gpd.clip(detection_gdf, aoi_gdf)

        full_name = os.path.basename(detection)
        file_name = os.path.splitext(full_name)
        year = (full_name.partition('year-'))[2][:4]

        if not detection_clip.empty:
            detection_clip.to_file(feature_path, layer='detections_' + year)
            written_files.append(feature_path) 

    print()
    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)
    print()