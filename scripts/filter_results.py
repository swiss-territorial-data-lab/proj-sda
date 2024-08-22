import argparse
import os
import sys
import time
import yaml

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from osgeo import gdal

sys.path.insert(0, '.')
from helpers import misc
from helpers.constants import DONE_MSG

from loguru import logger
logger = misc.format_logger(logger)


def calculate_slope(DEM):

    dir = os.path.dirname(DEM)
    gdal.DEMProcessing(dir + '/switzerland_slope.tif', DEM, 'slope')
    with rasterio.open(dir + '/switzerland_slope.tif') as dataset:
        slope = dataset.read(1)

    return slope


if __name__ == "__main__":

    # Chronometer
    tic = time.time()
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description="The script post-process the detections obtained with the object-detector")
    parser.add_argument('config_file', type=str, help='input geojson path')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")

    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    DETECTIONS = cfg['detections']
    TICINO_CAD = cfg['ticino']['cadastre']
    DEM = cfg['dem']
    SCORE = cfg['score']
    AREA = cfg['area']
    ELEVATION = cfg['elevation']
    SLOPE = cfg['slope']

    written_files = [] 

    # Convert input detections to a geodataframe 
    detections_gdf = gpd.read_file(DETECTIONS)
    detections_gdf = detections_gdf.to_crs(2056)
    detections_gdf = detections_gdf[detections_gdf['tag']!='FN']
    detections_gdf['area'] = detections_gdf.geometry.area 
    detections_gdf['det_id'] = detections_gdf.index
    total = len(detections_gdf)
    logger.info(f"{total} detections")

    ticino_cad_gdf = gpd.read_file(TICINO_CAD)
    ticino_cad_gdf = ticino_cad_gdf.to_crs(2056)

    # Discard polygons detected at/below 0 m and above the threshold elevation and above a given slope
    dem = rasterio.open(DEM)

    detections_gdf = detections_gdf.loc[detections_gdf['geometry'].is_valid, :] 
    row, col = dem.index(detections_gdf.centroid.x, detections_gdf.centroid.y)
    elevation = dem.read(1)[row, col]
    detections_gdf['elevation'] = elevation 

    logger.info("Slope computing")
    dem_slope = calculate_slope(DEM)
    slope = dem_slope[row, col]
    detections_gdf['slope'] = slope

    detections_gdf = detections_gdf[(detections_gdf.elevation != 0) & (detections_gdf.elevation < ELEVATION)]
    tdem = len(detections_gdf)
    logger.info(f"{total - tdem} detections were removed by elevation threshold: {ELEVATION} m")
    detections_gdf = detections_gdf[(detections_gdf.elevation != -9999) & (detections_gdf.slope <= SLOPE)]
    tslope = len(detections_gdf)
    logger.info(f"{tdem - tslope} detections were removed by slope threshold: {SLOPE}%")

    # Filter dataframe by score value
    detections_score = detections_gdf[detections_gdf.score > SCORE]
    sc = len(detections_score)
    logger.info(f"{tslope - sc} detections were removed by score filtering (score threshold = {SCORE})")

    # Discard polygons with area under the threshold 
    detections_area = detections_score[detections_score.area > AREA]
    ta = len(detections_area)
    logger.info(f"{sc - ta} detections were removed by area filtering (area threshold = {AREA} m2)")

    # Indicate if polygons are intersecting relevant vector layers with a min thd of 20% of the detection covered
    detections_area['det_id'] = detections_area.index
    detections_area['geometry'] = detections_area.geometry.buffer(-8)
    ticino_cad_gdf['cad_id'] = ticino_cad_gdf.index

    for gdf in [ticino_cad_gdf]: 
        detections_join = gpd.sjoin(detections_area, gdf, how='left', predicate='intersects')
        detections_join['test'] = np.where(detections_join.cad_id.isnull(), 'no', 'yes')
        # detections_filtered = detections_join[detections_join.cad_id.isnull()].copy()
    detections_join['geometry'] = detections_join.geometry.buffer(8)
    detections_filtered = detections_join.drop(columns=['index_right', 'NAME'])
    detections_filtered = detections_join.drop_duplicates(subset='det_id')

    # Final gdf
    logger.info(f"{len(detections_filtered)} detections remaining after filtering")

    # Formatting the output name of the filtered detection  
    feature = f'{DETECTIONS[:-5]}_threshold_score-{SCORE}_area-{int(AREA)}_elevation-{int(ELEVATION)}_slope-{int(SLOPE)}'.replace('0.', '0dot') + '.gpkg'
    detections_filtered.to_file(feature)

    written_files.append(feature)
    logger.success(f"{DONE_MSG} A file was written: {feature}")  

    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()