import argparse
import os
import sys
import time
import yaml
from glob import glob

import geopandas as gpd
import numpy as np
import pandas as pd

import json
from hashlib import md5

sys.path.insert(0, '.')
import functions.graphs as graphs
import functions.metrics as metrics
import functions.misc as misc
from functions.constants import DONE_MSG

from loguru import logger
logger = misc.format_logger(logger)


def group_detections(detections_gdf, IoU_threshold, ignore_year=False):
    """
    Mark detections representing the same object as a group.

    Given a GeoDataFrame of detections, find the detections that overlap
    with each other and assign them the same group_id. This is done by
    computing the intersection over union of each pair of detections and
    then grouping the detections using a graph algorithm.

    Parameters
    ----------
    detections_gdf : GeoDataFrame
        A GeoDataFrame containing the detections to be grouped.
    IoU_threshold : float
        The minimum intersection over union required for two detections to
        be considered as overlapping.
    ignore_year : bool
        If True, ignore the year condition when computing the IoU.

    Returns
    -------
    new_dets_gdf : GeoDataFrame
        A GeoDataFrame containing the same information as the input
        detections_gdf but with an additional column 'group_id' that
        assigns a unique group_id to each group of overlapping detections.
    """
    
    overlap_detections_gdf = self_intersect(detections_gdf, ignore_year)
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


def self_intersect(gdf, ignore_year=False):
    _gdf = gdf.copy()
    _gdf['geom'] = _gdf.geometry

    overlap_detections_gdf = gpd.sjoin(_gdf, _gdf, how='inner')
    overlap_detections_gdf = overlap_detections_gdf[
        (overlap_detections_gdf.merged_id_left>=overlap_detections_gdf.merged_id_right) # We keep self-intersection, so that alone dets don't have a presence of 0.
        # & (overlap_detections_gdf.det_category_left==overlap_detections_gdf.det_category_right)
        & (True if ignore_year else overlap_detections_gdf.year_det_left==overlap_detections_gdf.year_det_right)
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
    ASSESS = cfg['assess']['enable']
    if ASSESS:
        METHOD = cfg['assess']['metrics_method']
        LABELS = cfg['labels'] if 'labels' in cfg.keys() else None
        CATEGORIES = cfg['categories']

    written_files = []
    os.chdir(WORKING_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info('Read data...')
    detections_list = glob("*/**/merged_detections_at_0dot05_threshold.gpkg", recursive=True)

    nbr_dets_list = []
    detections_gdf = gpd.GeoDataFrame()
    for file in detections_list:
        logger.info(f"Reading {file}")
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
    groups_df = intersecting_detections_gdf['group_id'].value_counts().reset_index().rename(columns={'count': 'count_dets'})
    groups_df['percentage'] = groups_df['count_dets']/len(detections_list)

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
    logger.success(f'The covered area is {round(groupped_detections_gdf.unary_union.area/1000000)} km2.')

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
            final_groupped_dets_gdf.loc[final_groupped_dets_gdf.merged_id==detection.merged_id_top, 'merged_score'] = min(det_score + 0.1, 1)

    excessive_columns = ['wkb_geom', 'merged_id_bottom', 'merged_id_top']
    filepath = os.path.join(OUTPUT_DIR, 'groupped_detections.gpkg')
    final_groupped_dets_gdf.drop(columns=excessive_columns).to_file(filepath, crs='EPSG:2056', driver='GPKG', index=False)
    written_files.append(filepath)

    filepath = os.path.join(OUTPUT_DIR, 'removed_detections.gpkg')
    removed_dets_gdf.drop(columns=excessive_columns).to_file(filepath, crs='EPSG:2056', driver='GPKG', index=False)
    written_files.append(filepath)

    del overlap_detections_gdf, intersecting_dets_gdf, filtered_groupped_dets_gdf

    logger.info('Merge overlapping components of the same year...')
    dets_to_merge_gdf = final_groupped_dets_gdf.drop(columns='group_id')
    dets_to_merge_gdf.loc[:, 'geometry'] = dets_to_merge_gdf.buffer(5)
    intersecting_detections_gdf = group_detections(dets_to_merge_gdf, 0.01)
    intersecting_detections_gdf.drop(columns=['second_IoU'], inplace=True)
    intersecting_detections_gdf.sort_values('merged_score', ascending=False, inplace=True)
    merged_detections_gdf = intersecting_detections_gdf.dissolve('group_id', aggfunc='first', as_index=False)
    merged_detections_gdf.loc[:, 'geometry'] = merged_detections_gdf.buffer(-5)
    merged_detections_gdf.set_crs(2056, inplace=True)

    filepath = os.path.join(OUTPUT_DIR, 'merged_detections.gpkg')
    merged_detections_gdf.drop(columns=excessive_columns).to_file(filepath, crs='EPSG:2056', index=False)
    written_files.append(filepath)

    logger.success(f"{len(final_groupped_dets_gdf)} features were kept.")
    logger.success(f"Once dissolved, {len(merged_detections_gdf)} features are left")
    logger.success(f'The covered area is {round(merged_detections_gdf.unary_union.area/1000000)} km2.')
    logger.success(f"{len(removed_dets_gdf)} features were removed.")

    if ASSESS:
        logger.info("Loading labels as a GeoPandas DataFrame...")
        labels_gdf = gpd.read_file(LABELS)
        labels_gdf = labels_gdf.to_crs(2056)
        if 'year' in labels_gdf.keys():  
            labels_gdf['year'] = labels_gdf.year.astype(int)       
            labels_gdf = labels_gdf.rename(columns={"year": "year_label"})
        logger.success(f"{DONE_MSG} {len(labels_gdf)} features were found.")

        # get classe ids
        categories_info_df, id_classes = misc.get_categories(CATEGORIES)

        # append class ids to labels
        labels_gdf['CATEGORY'] = labels_gdf.CATEGORY.astype(str)
        labels_w_id_gdf = labels_gdf.merge(categories_info_df, on='CATEGORY', how='left')

        logger.info('Tag detections and get metrics...')

        metrics_dict = {}
        metrics_dict_by_cl = []
        metrics_cl_df_dict = {}

        tp_gdf, fp_gdf, fn_gdf, mismatched_class_gdf, small_poly_gdf = metrics.get_fractional_sets(
            merged_detections_gdf.drop(columns=excessive_columns), labels_w_id_gdf, 0.1, 0.05)

        tp_gdf['tag'] = 'TP'
        fp_gdf['tag'] = 'FP'
        fn_gdf['tag'] = 'FN'
        mismatched_class_gdf['tag'] = 'wrong class'
        small_poly_gdf['tag'] = 'small polygon'

        tp_k, fp_k, fn_k, p_k, r_k, f1_k, accuracy, precision, recall, f1 = metrics.get_metrics(tp_gdf, fp_gdf, fn_gdf, mismatched_class_gdf, id_classes, method=METHOD)
        logger.info(f'Detection score threshold = 0.05')
        logger.info(f'accuracy = {accuracy:.3f}')
        logger.info(f'Method = {METHOD}: precision = {precision:.3f}, recall = {recall:.3f}, f1 = {f1:.3f}')

        tagged_dets_gdf = pd.concat([tp_gdf, fp_gdf, fn_gdf, mismatched_class_gdf], ignore_index=True)

        logger.info(f'TP = {len(tp_gdf)}, FP = {len(fp_gdf)}, FN = {len(fn_gdf)}')
        tagged_dets_gdf['det_category'] = [
            categories_info_df.loc[categories_info_df.label_class==det_class+1, 'CATEGORY'].iloc[0] 
            if not np.isnan(det_class) else None
            for det_class in tagged_dets_gdf.det_class.to_numpy()
        ] 

        # Save tagged processed results 
        feature = os.path.join(OUTPUT_DIR, f'tagged_merged_detections_at_0dot05_threshold.gpkg'.replace('0.', '0dot'))
        tagged_dets_gdf = tagged_dets_gdf.to_crs(2056)
        tagged_dets_gdf = tagged_dets_gdf.rename(columns={'CATEGORY': 'label_category'}, errors='raise')
        tagged_dets_gdf[['geometry', 'det_id', 'score', 'tag', 'label_class', 'label_category', 'year_label', 'det_class', 'det_category', 'year_det']]\
            .to_file(feature, driver='GPKG', index=False)
        written_files.append(feature)

    logger.info('Merge overlapping components while ignoring the year...')
    dets_to_merge_gdf = merged_detections_gdf.drop(columns='group_id')
    dets_to_merge_gdf.loc[:, 'geometry'] = dets_to_merge_gdf.buffer(5)
    intersecting_detections_gdf = group_detections(dets_to_merge_gdf, 0.5, ignore_year=True)
    intersecting_detections_gdf.sort_values('merged_score', ascending=False, inplace=True)
    intersecting_detections_gdf.drop(columns=excessive_columns, inplace=True)

    merged_dets_across_years_gdf = intersecting_detections_gdf.dissolve('group_id', aggfunc='first', as_index=False)
    min_year_df = intersecting_detections_gdf[['group_id', 'year_det']].groupby('group_id', as_index=False).min().rename(columns={'year_det': 'first_year'})
    max_year_df = intersecting_detections_gdf[['group_id', 'year_det']].groupby('group_id', as_index=False).max().rename(columns={'year_det': 'last_year'})
    count_year_df = intersecting_detections_gdf.group_id.value_counts().reset_index().rename(columns={'count': 'count_years'})

    for df in  [min_year_df, max_year_df, count_year_df]:
        merged_dets_across_years_gdf = pd.merge(merged_dets_across_years_gdf, df, on='group_id', how='left')
    merged_dets_across_years_gdf.loc[:, 'merged_score'] = merged_dets_across_years_gdf.apply(
        lambda x: min(1, x['merged_score'] + 0.1 * (x['count_years']-1)), axis=1
    )

    merged_dets_across_years_gdf.loc[:, 'geometry'] = merged_dets_across_years_gdf.buffer(-5)
    merged_dets_across_years_gdf.set_crs(2056, inplace=True)

    filepath = os.path.join(OUTPUT_DIR, 'merged_detections_across_years.gpkg')
    merged_dets_across_years_gdf.to_file(filepath, crs='EPSG:2056', index=False)
    written_files.append(filepath)

    logger.success(f"{DONE_MSG} {len(merged_dets_across_years_gdf)} features were left after merging across years.")    

    logger.info('The following files were written:')
    for written_file in written_files:
        logger.info(written_file)

    # Chronometer
    toc = time.time()
    logger.info(f"Done in {toc - tic:.2f} seconds.")