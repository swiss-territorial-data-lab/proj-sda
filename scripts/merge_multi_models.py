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


def group_detections(detections_gdf, IoU_threshold):
    
    overlap_detections_gdf = self_intersect(detections_gdf)
    geom1 = overlap_detections_gdf.geom_left.tolist()
    geom2 = overlap_detections_gdf.geom_right.tolist()
    iou = []
    for (i, ii) in zip(geom1, geom2):
        iou.append(metrics.intersection_over_union(i, ii))
    overlap_detections_gdf['IoU'] = iou
    overlap_detections_gdf = overlap_detections_gdf[overlap_detections_gdf['IoU']>=IoU_threshold]

    logger.info('Mark detections representing the same object as a group...')
    groups = graphs.make_groups(overlap_detections_gdf, 'wkb_geom_left', 'wkb_geom_right')
    overlap_detections_gdf = overlap_detections_gdf.apply(lambda x: graphs.assign_groups(x, groups, 'wkb_geom_left'), axis=1)
    intersecting_detections_gdf = pd.concat([
        overlap_detections_gdf.drop_duplicates(subset=['merged_id_left', 'group_id']).rename(columns={'merged_id_left':'merged_id'}),
        overlap_detections_gdf.drop_duplicates(subset=['merged_id_right', 'group_id']).rename(columns={'merged_id_right':'merged_id'})
    ])
    intersecting_detections_gdf.drop_duplicates(subset=['merged_id'], inplace=True)
    new_dets_gdf = pd.merge(detections_gdf, intersecting_detections_gdf[['merged_id', 'group_id']], how='left', on='merged_id')

    return new_dets_gdf


def self_intersect(gdf):
    _gdf = gdf.copy()
    _gdf['geom'] = _gdf.geometry

    overlap_detections_gdf = gpd.sjoin(_gdf, _gdf, how='inner')
    overlap_detections_gdf = overlap_detections_gdf[
        (overlap_detections_gdf.merged_id_left>=overlap_detections_gdf.merged_id_right) # We keep self-intersection, so that alone dets don't have a presence of 0.
        # & (overlap_detections_gdf.det_category_left==overlap_detections_gdf.det_category_right)
        & (overlap_detections_gdf.year_det_left==overlap_detections_gdf.year_det_right)
    ].copy()

    return overlap_detections_gdf


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

    THRESHOLD = cfg['threshold']

    written_files = []
    os.chdir(WORKING_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info('Read data...')
    detections_list = glob("*/**/merged_detections_at_0dot05_threshold.gpkg", recursive=True)

    nbr_dets_list = []
    detections_gdf = gpd.GeoDataFrame()
    for file in detections_list:
        model_name = os.path.dirname(file)
        tmp_gdf = gpd.read_file(file)
        tmp_gdf['model'] = os.path.dirname(file)    
        detections_gdf = pd.concat([detections_gdf, tmp_gdf], ignore_index=True)      # If prblm with Multipoly, use spatial index

        nbr_dets_list.append(len(tmp_gdf))

    logger.info(f"{DONE_MSG} {sum(nbr_dets_list)/len(nbr_dets_list):.2f} features were found in average for each model.")
    
    detections_gdf['merged_id'] = detections_gdf.index
    encoded_geoms = detections_gdf.geometry.to_wkb()
    detections_gdf['wkb_geom'] = encoded_geoms.apply(lambda x: md5(x))
    
    logger.info('Intersect detections...')
    # Condsider a det to be the same object if more than 50% the same surface
    intersecting_detections_gdf = group_detections(detections_gdf, 0.5)

    logger.info('Count number of dets in each group...')
    groups_df = intersecting_detections_gdf['group_id'].value_counts().reset_index()
    groups_df['percentage'] = groups_df['count']/len(detections_list)

    # Bring group info back to the detections
    completed_detections_gdf = pd.merge(intersecting_detections_gdf, groups_df, how='left', left_on='group_id', right_on='group_id')
    completed_detections_gdf.loc[completed_detections_gdf['group_id'].isna(), 'percentage'] = 0     # Currently, no det is removed before groupping
    completed_detections_gdf['merged_score'] = completed_detections_gdf['score'] * completed_detections_gdf['percentage']

    filepath = os.path.join(OUTPUT_DIR, 'all_detections.gpkg')
    completed_detections_gdf.drop(columns='wkb_geom').to_file(filepath, driver='GPKG', index=False)
    written_files.append(filepath)
    del detections_gdf, intersecting_detections_gdf,

    completed_detections_gdf.sort_values('score', inplace=True, ascending=False)
    groupped_detections_gdf = completed_detections_gdf.groupby('group_id', as_index=False).agg('first')

    logger.success(f'{DONE_MSG} Once groupped, there are {len(groupped_detections_gdf)} detections.')
    logger.success(f'The covered area is {round(groupped_detections_gdf.unary_union.area)}.')

    logger.info('Remove detection based on score...')
    condition_to_keep = (groupped_detections_gdf.merged_score > THRESHOLD) \
        & ((groupped_detections_gdf.merged_score > 0.2) | (groupped_detections_gdf.area < 100000))
    filtered_groupped_dets_gdf = groupped_detections_gdf[condition_to_keep]
    removed_dets_gdf = groupped_detections_gdf[~condition_to_keep]

    logger.info('Remove detections inside other detections...')
    overlap_detections_gdf = self_intersect(filtered_groupped_dets_gdf[['geometry', 'merged_id', 'year_det', 'det_category']])
    overlap_detections_gdf = overlap_detections_gdf[overlap_detections_gdf.merged_id_left!=overlap_detections_gdf.merged_id_right]  # Remove self-intersection
    intersected_geoms = overlap_detections_gdf.geom_left.intersection(overlap_detections_gdf.geom_right)
    # Test if left geom is into right geom
    det_geom = overlap_detections_gdf.geom_left.tolist()
    iou = []
    for (i, ii) in zip(intersected_geoms, det_geom):
        iou.append(metrics.intersection_over_union(i, ii))
    overlap_detections_gdf['IoU_left'] = iou
    # Test if right geom is into left geom
    det_geom = overlap_detections_gdf.geom_right.tolist()
    iou = []
    for (i, ii) in zip(intersected_geoms, det_geom):
        iou.append(metrics.intersection_over_union(i, ii))
    overlap_detections_gdf['IoU_right'] = iou

    intersecting_dets_gdf = pd.concat(
        [
            overlap_detections_gdf[['merged_id_left', 'IoU_left', 'merged_id_right']].groupby(['merged_id_left'], as_index=False).max().rename(
                columns={'merged_id_left':'merged_id_bottom', 'IoU_left':'second_IoU', 'merged_id_right':'merged_id_top'}
            ), 
            overlap_detections_gdf[['merged_id_right', 'IoU_right', 'merged_id_left']].groupby(['merged_id_right'], as_index=False).max().rename(
                columns={'merged_id_right':'merged_id_bottom', 'IoU_right':'second_IoU', 'merged_id_left':'merged_id_top'}
            )
        ],
        ignore_index=True
    )
    intersecting_dets_gdf.sort_values('second_IoU', inplace=True, ascending=False)
    intersecting_dets_gdf.drop_duplicates(subset='merged_id_bottom', inplace=True)

    # Remove dets under another det
    filtered_groupped_dets_gdf = pd.merge(filtered_groupped_dets_gdf, intersecting_dets_gdf, how='left', left_on='merged_id', right_on='merged_id_bottom')
    condition = (filtered_groupped_dets_gdf.second_IoU<0.9) | filtered_groupped_dets_gdf.second_IoU.isna()
    final_groupped_dets_gdf = filtered_groupped_dets_gdf[condition]
    removed_dets_gdf = pd.concat([removed_dets_gdf, filtered_groupped_dets_gdf[~condition]], ignore_index=True)

    # Give a bonus to a det if there was another highly present det under it
    intersecting_dets_gdf = intersecting_dets_gdf[
        intersecting_dets_gdf.merged_id_top.isin(final_groupped_dets_gdf.merged_id.unique().tolist())
        & intersecting_dets_gdf.merged_id_bottom.isin(removed_dets_gdf.merged_id.unique().tolist())
    ] 
    for detection in intersecting_dets_gdf.itertuples():
        if removed_dets_gdf.loc[removed_dets_gdf.merged_id==detection.merged_id_bottom, 'percentage'].iloc[0] >= 0.75:
            det_score = final_groupped_dets_gdf.loc[final_groupped_dets_gdf.merged_id==detection.merged_id_top, 'merged_score'].iloc[0]
            final_groupped_dets_gdf.loc[final_groupped_dets_gdf.merged_id==detection.merged_id_top, 'merged_score'] = det_score + 0.1 if det_score < 0.9 else det_score

    excessive_columns = ['wkb_geom', 'merged_id_bottom', 'merged_id_top']
    filepath = os.path.join(OUTPUT_DIR, 'groupped_detections.gpkg')
    final_groupped_dets_gdf.drop(columns=excessive_columns).to_file(filepath, crs='EPSG:2056', driver='GPKG', index=False)
    written_files.append(filepath)

    filepath = os.path.join(OUTPUT_DIR, 'removed_detections.gpkg')
    removed_dets_gdf.drop(columns=excessive_columns).to_file(filepath, crs='EPSG:2056', driver='GPKG', index=False)
    written_files.append(filepath)

    del overlap_detections_gdf, intersecting_dets_gdf, filtered_groupped_dets_gdf

    logger.info('Merge overlapping components...')
    dets_to_merge_gdf = final_groupped_dets_gdf.drop(columns='group_id')
    dets_to_merge_gdf.loc[:, 'geometry'] = dets_to_merge_gdf.buffer(1)
    intersecting_detections_gdf = group_detections(dets_to_merge_gdf, 0.01)
    intersecting_detections_gdf.drop(columns=excessive_columns+['second_IoU'], inplace=True)
    intersecting_detections_gdf.sort_values('merged_score', ascending=False, inplace=True)
    merged_detections_gdf = intersecting_detections_gdf.dissolve('group_id', aggfunc='first', as_index=False)
    merged_detections_gdf.loc[:, 'geometry'] = merged_detections_gdf.buffer(-1)

    filepath = os.path.join(OUTPUT_DIR, 'merged_detections.gpkg')
    merged_detections_gdf.to_file(filepath, crs='EPSG:2056', index=False)
    written_files.append(filepath)

    logger.success(f"{DONE_MSG} {len(final_groupped_dets_gdf)} features were kept.")
    logger.success(f"Once dissolved, {len(merged_detections_gdf)} features are left")
    logger.success(f'The covered area is {round(merged_detections_gdf.unary_union.area)}.')
    logger.success(f"{len(removed_dets_gdf)} features were removed.")

    logger.info('The following files were written:')
    for written_file in written_files:
        logger.info(written_file)

    # Chronometer
    toc = time.time()
    logger.info(f"Done in {toc - tic:.2f} seconds.")