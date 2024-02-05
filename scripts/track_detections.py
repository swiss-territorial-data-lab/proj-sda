import time
import argparse
import yaml
import os, sys, inspect
import geopandas as gpd
import pandas as pd
from loguru import logger

# the following allows us to import modules from within this file's parent folder
sys.path.insert(1, '.')

logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}", level="INFO")

if __name__ == "__main__":

    # Start chronometer
    tic = time.time()
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description="Track detections in a muli-year dataset (STDL.proj-sda)")
    parser.add_argument('config_file', type=str, help='Framework configuration file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")
 
    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    WORKING_DIR = cfg['working_dir']
    YEARS = sorted(cfg['years'])
    DETECTIONS_SHP = cfg['detections_shapefile']
    OUTPUT_DIR = cfg['output_dir']

    os.chdir(WORKING_DIR)

    # Create an output directory in case it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    written_files = []

    # Concatenate all the dataframe obtained for a given year to a single dataframe
    print()
    logger.info(f"Concaneting different years dataframes:")
    for YEAR in YEARS:      
        if YEAR == YEARS[0]: 
            gdf = gpd.read_file(DETECTIONS_SHP, layer='detections_' + str(YEAR))
            gdf['year'] = YEAR 
            logger.info(f" Year: {YEAR} - {len(gdf)} detections")
        else:
            gdfx = gpd.read_file(DETECTIONS_SHP, layer='detections_' + str(YEAR))
            gdfx['year'] = YEAR 
            logger.info(f" Year: {YEAR} - {len(gdfx)} detections")
            gdf = pd.concat([gdf, gdfx])

    print(gdf)

    # Create a dataframe with merged overlapping polygons
    print()
    logger.info(f"Merging overlapping polygons")
    gdf_all = gdf.geometry.unary_union
    gdf_all = gpd.GeoDataFrame(geometry=[gdf_all], crs='EPSG:2056')  
    logger.info(f"Attribute unique object id")
    gdf_all = gdf_all.explode(index_parts = True).reset_index(drop=True)
    labels = gdf_all.index

    # Spatially compare the global dataframe with all detections by year and the merged dataframe. Allow to attribute a unique ID to each detection
    print()
    logger.info(f"Compare single polygon dataframes detection to merge polygons dataframe")
    intersection = gpd.sjoin(gdf, gdf_all, how='inner')

    # Reorganised dataframe columns and save files 
    print()
    logger.info(f"Save files")
    intersection.rename(columns={'index_right': 'id_object'}, inplace=True)
    gdf_final = intersection[['id_object', 'id_feature', 'year', 'score', 'area', 'centroid_x', 'centroid_y', 'geometry']]
    feature_path = os.path.join(OUTPUT_DIR, 'detections_years.gpkg')
    gdf_final.to_file(feature_path) 
    written_files.append(feature_path) 

    feature_path = os.path.join(OUTPUT_DIR, 'detections_years.csv')
    gdf_final.to_csv(feature_path, index=False)
    written_files.append(feature_path) 

    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)
    print()

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()