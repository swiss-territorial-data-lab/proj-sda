#!/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import yaml

import geopandas as gpd

sys.path.insert(0, '.')
import functions.fct_misc as misc
from functions.constants import DONE_MSG

from loguru import logger
logger = misc.format_logger(logger)


def main(WORKING_DIR, CANTON, YEAR, SRS, CANTON_SHP, IMG_FOOTPRINT_SHP):

    os.chdir(WORKING_DIR)
    logger.info(f'Working directory set to {WORKING_DIR}')

    written_files = []
    
    # Read shapefiles
    logger.info("Loading canton border.")
    canton_aoi_gdf = gpd.read_file(CANTON_SHP)
    canton_aoi_gdf = misc.convert_crs(canton_aoi_gdf, epsg=SRS)
    logger.info("Loading images footprint.")
    img_fp_gdf = gpd.read_file(IMG_FOOTPRINT_SHP)
    img_fp_gdf = misc.convert_crs(img_fp_gdf, epsg=SRS)
    img_fp_gdf['flight_year'] = YEAR

    # Intersect shapefiles
    aoi_intersection_gdf = canton_aoi_gdf.overlay(img_fp_gdf, how='intersection')
    aoi_intersection_gdf = aoi_intersection_gdf[['geometry', 'flight_year']] 
    aoi_intersection_gdf['canton'] = CANTON
    aoi_intersection_gdf['shp1'] = os.path.basename(CANTON_SHP)
    aoi_intersection_gdf['shp2'] = os.path.basename(IMG_FOOTPRINT_SHP)

    # Save shapefile
    if len(aoi_intersection_gdf) > 0:
        feature = f'{CANTON}/aoi_{YEAR}_{CANTON}.gpkg'
        aoi_intersection_gdf.to_file(feature)
        written_files.append(feature) 
        logger.success(f"{DONE_MSG} A file was written: {feature}") 
    else:
        logger.info("The 2 shapefiles do not intersect. No file is saved.")    

    print()
    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)
    print()


if __name__ == "__main__":

    # Start chronometer
    tic = time.time()
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description="The script prepares the AoI used to perform inference")
    parser.add_argument('config_file', type=str, help='Framework configuration file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")
 
    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    WORKING_DIR = cfg['working_directory']
    CANTON = cfg['canton']
    YEAR = cfg['year']
    SRS = cfg['srs']
    CANTON_SHP = cfg['canton_shp'].replace('{canton}', CANTON)
    IMG_FOOTPRINT_SHP = cfg['img_footprint_shp'].replace('{year}', str(YEAR))

    main(WORKING_DIR, CANTON, YEAR, SRS, CANTON_SHP, IMG_FOOTPRINT_SHP)

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")
    
    sys.stderr.flush()