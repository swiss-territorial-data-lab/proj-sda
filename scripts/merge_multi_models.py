import argparse
import os
import sys
import time
import yaml
from glob import glob

import geopandas as gpd
import numpy as np
import pandas as pd

from hashlib import md5

sys.path.insert(0, '.')
import functions.fct_graphs as graphs
import functions.fct_metrics as metrics
import functions.fct_misc as misc
from functions.constants import DONE_MSG

from loguru import logger
logger = misc.format_logger(logger)


if __name__ == "__main__":

    # Chronometer
    tic = time.time()
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description="The script tests the fusion of the results from multiple models to assess the proba of FP.")
    parser.add_argument('config_file', type=str, help='input geojson path')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")

    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    WORKING_DIR = cfg['working_dir']
    OUTPUT_DIR = cfg['output_dir']
    written_files = []

    os.chdir(WORKING_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info('Read data...')
    detections_list = glob("*/**/merged_detections_at_0dot05_threshold.gpkg", recursive=True)

    detections_dict = {}
    detections_gdf = gpd.GeoDataFrame()
    for file in detections_list:
        model_name = os.path.dirname(file)
        tmp_gdf = gpd.read_file(file)
        tmp_gdf['model'] = os.path.dirname(file)    
        detections_gdf = pd.concat([detections_gdf, tmp_gdf], ignore_index=True)      # If prblm with Multipoly, use spatial index
    
    detections_gdf['merged_id'] = detections_gdf.index
    encoded_geoms = detections_gdf.geometry.to_wkb()
    detections_gdf['wkb_geom'] = encoded_geoms.apply(lambda x: md5(x))
    
    logger.info('Intersect detections...')
    # Condsider a det to be the same object if more than 50% the same surface
    detections_gdf['geom'] = detections_gdf.geometry
    overlap_detections_gdf = gpd.sjoin(detections_gdf, detections_gdf, how='inner')
    overlap_detections_gdf = overlap_detections_gdf[
        (overlap_detections_gdf.merged_id_left>=overlap_detections_gdf.merged_id_right)
        & (overlap_detections_gdf.det_category_left==overlap_detections_gdf.det_category_right)
        & (overlap_detections_gdf.year_det_left==overlap_detections_gdf.year_det_right)
    ].copy()
    geom1 = overlap_detections_gdf.geom_left.tolist()
    geom2 = overlap_detections_gdf.geom_right.tolist()
    iou = []
    for (i, ii) in zip(geom1, geom2):
        iou.append(metrics.intersection_over_union(i, ii))
    overlap_detections_gdf['IoU'] = iou
    overlap_detections_gdf = overlap_detections_gdf[overlap_detections_gdf['IoU']>=0.5]

    # Mark detections representing the same object as a group 
    groups = graphs.make_groups(overlap_detections_gdf, 'wkb_geom_left', 'wkb_geom_right')
    overlap_detections_gdf = overlap_detections_gdf.apply(lambda x: graphs.assign_groups(x, groups, 'wkb_geom_left'), axis=1)
    intersecting_detections_df = pd.concat([
        overlap_detections_gdf.drop_duplicates(subset=['merged_id_left', 'group_id']).rename(columns={'merged_id_left':'merged_id'}),
        overlap_detections_gdf.drop_duplicates(subset=['merged_id_right', 'group_id']).rename(columns={'merged_id_right':'merged_id'})
    ])
    intersecting_detections_df.drop_duplicates(subset=['merged_id'], inplace=True)


    # Count number of dets in each group
    groups_df = intersecting_detections_df['group_id'].value_counts().reset_index()
    groups_df['percentage'] = groups_df['count']/len(detections_list)

    # Bring group info back to the detections
    tmp_gdf = pd.merge(detections_gdf, intersecting_detections_df[['merged_id', 'group_id']], how='left', on='merged_id')
    completed_detections_gdf = pd.merge(tmp_gdf, groups_df, how='left', left_on='group_id', right_on='group_id')
    completed_detections_gdf.drop(columns=['wkb_geom', 'geom'], inplace=True)
    completed_detections_gdf.loc[completed_detections_gdf['group_id'].isna(), 'percentage'] = 0

    filepath = os.path.join(OUTPUT_DIR, 'all_detections.gpkg')
    completed_detections_gdf.to_file(filepath, driver='GPKG', index=False)
    written_files.append(filepath)

    completed_detections_gdf.sort_values('score', inplace=True, ascending=False)
    groupped_detections_gdf = completed_detections_gdf.groupby('group_id', as_index=False).agg('first')

    filepath = os.path.join(OUTPUT_DIR, 'groupped_detections.gpkg')
    groupped_detections_gdf.to_file(filepath, driver='GPKG', index=False)
    written_files.append(filepath)

    logger.success(f"{DONE_MSG} {len(groupped_detections_gdf)} features were found.")

    logger.info('The following files were written:')
    for written_file in written_files:
        logger.info(written_file)

    # Chronometer
    toc = time.time()
    logger.info(f"Done in {toc - tic:.2f} seconds.")