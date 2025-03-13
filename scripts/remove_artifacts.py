import argparse
import os
import sys
import time
import yaml
from tqdm import tqdm

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, '.')
import functions.metrics as metrics
import functions.misc as misc
from functions.constants import DONE_MSG

from loguru import logger
logger = misc.format_logger(logger)


if __name__ == "__main__":

    # Chronometer
    tic = time.time()
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description="The script removes artifacts by comparing dets and tiles.")
    parser.add_argument('config_file', type=str, help='input geojson path')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")

    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    WORKING_DIR = cfg['working_directory']
    DETECTIONS = cfg['detections']
    TILES = cfg['tiles']

    CLOSE_AREA_MIN = cfg['close_area_min']
    CLOSE_AREA_MAX = cfg['close_area_max']
    IOU_THRESHOLD = cfg['iou_threshold']
    MIN_ARTIFACT_AREA = 500
    BUFFER = 10

    os.chdir(WORKING_DIR)
    OUTPUT_DIR = os.path.dirname(DETECTIONS)
    written_files = []

    logger.info('Read data...')

    detections_gdf = gpd.read_file(DETECTIONS)
    ID = 'merged_id' if 'merged_id' in detections_gdf.columns else 'det_id'
    detections_gdf['area'] = round(detections_gdf.area)
    nbr_dets = len(detections_gdf)
    logger.info(f"{nbr_dets} detections")

    tiles_gdf = gpd.read_file(TILES)
    tiles_gdf.to_crs(detections_gdf.crs, inplace=True)
    logger.info(f"{len(tiles_gdf)} tiles")

    logger.info('Filter non-artifact data...')
    logger.info(f'The detections with an area between {CLOSE_AREA_MIN} and {CLOSE_AREA_MAX} m2 are considered.')

    condition = (detections_gdf.area > CLOSE_AREA_MIN) & (detections_gdf.area < CLOSE_AREA_MAX) & (detections_gdf.count_years <=2)
    pot_artifact_dets_gdf = detections_gdf[condition].copy()
    non_artifact_dets_gdf = detections_gdf[~condition].copy()
    del detections_gdf

    tiles_gdf['tile_geom'] = tiles_gdf.geometry
    intersecting_dets_tiles_gdf = gpd.sjoin(pot_artifact_dets_gdf, tiles_gdf[['id', 'geometry', 'tile_geom']], how='left')
    assert not intersecting_dets_tiles_gdf.id.isna().any(), "Some detections were not joined with any tile."

    logger.info('Remove artifacts...')
    iou = []
    for (i, ii) in zip(intersecting_dets_tiles_gdf.geometry, intersecting_dets_tiles_gdf.tile_geom):
        iou.append(metrics.intersection_over_union(i, ii))
    intersecting_dets_tiles_gdf['IoU'] = iou
    sorted_intersecting_dets_tiles_gdf = intersecting_dets_tiles_gdf.sort_values('IoU', ascending=False).drop_duplicates([ID], ignore_index=True).round({'IoU': 2})

    assert sorted_intersecting_dets_tiles_gdf.IoU.max() <= 1, \
        f"Incoherent IoU over 1 for {sorted_intersecting_dets_tiles_gdf[sorted_intersecting_dets_tiles_gdf.IoU>1].shape[0]} detections."

    condition = sorted_intersecting_dets_tiles_gdf['IoU'] >= IOU_THRESHOLD
    artifacts_gdf = sorted_intersecting_dets_tiles_gdf[condition].drop(columns='tile_geom')
    non_artifact_dets_gdf = pd.concat([
        non_artifact_dets_gdf, sorted_intersecting_dets_tiles_gdf.loc[~condition, non_artifact_dets_gdf.columns]
    ], ignore_index=True)
    assert nbr_dets == len(artifacts_gdf) + len(non_artifact_dets_gdf), "Tiles went missing when identifying square artifacts."
    del pot_artifact_dets_gdf, intersecting_dets_tiles_gdf, sorted_intersecting_dets_tiles_gdf

    logger.success(f"{DONE_MSG} {len(artifacts_gdf)} detections were removed as artifacts.")
    if False:
        logger.info(f"Try to recreate sharp angles on detections...")
        condition = non_artifact_dets_gdf.area > MIN_ARTIFACT_AREA
        pot_artifact_dets_gdf = non_artifact_dets_gdf[condition].copy()

        intersecting_dets_tiles_gdf = gpd.sjoin(pot_artifact_dets_gdf, tiles_gdf[['id', 'geometry', 'tile_geom']])
        duplicated_ids = intersecting_dets_tiles_gdf.loc[intersecting_dets_tiles_gdf.duplicated(ID), ID].unique()
        intersecting_dets_tiles_gdf = intersecting_dets_tiles_gdf[~intersecting_dets_tiles_gdf[ID].isin(duplicated_ids)]

        intersecting_dets_tiles_gdf.loc[:, 'geometry'] = intersecting_dets_tiles_gdf.buffer(20)
        intersecting_dets_tiles_gdf.loc[:, 'geometry'] = [det.geometry.intersection(det.tile_geom) for det in intersecting_dets_tiles_gdf.itertuples()]
        sharp_dets_gdf = intersecting_dets_tiles_gdf[non_artifact_dets_gdf.columns].reset_index(drop=True)

        filepath = os.path.join(OUTPUT_DIR, 'test_angles.gpkg')
        sharp_dets_gdf.to_file(filepath)
        written_files.append(filepath)

    logger.info("Export results...")

    filepath = os.path.join(OUTPUT_DIR, f'artifacts_A-{CLOSE_AREA_MIN}-{CLOSE_AREA_MAX}_IoU-{IOU_THRESHOLD}.gpkg')
    artifacts_gdf.to_file(filepath, index=False)
    written_files.append(filepath)

    filepath = os.path.join(OUTPUT_DIR, 'actual_dets_across_years.gpkg')
    non_artifact_dets_gdf.to_file(filepath)
    written_files.append(filepath)

    logger.info('The following files were written:')
    for written_file in written_files:
        logger.info(written_file)

    # Chronometer
    toc = time.time()
    logger.info(f"Done in {toc - tic:.2f} seconds.")