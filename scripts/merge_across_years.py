import argparse
import os
import time
import sys
import yaml
from loguru import logger

import geopandas as gpd
import pandas as pd

from hashlib import md5

sys.path.insert(0, '.')
from functions.constants import DONE_MSG, OVERWRITE
from functions.metrics import perform_assessment
from functions.misc import format_logger, get_categories
from merge_multi_results import group_detections

logger = format_logger(logger)

def main(all_years_dets_gdf, assess=False, method=None, labels_path=None, categories_path=None, output_dir='output'):
    written_files = []

    last_written_file = os.path.join(output_dir, 'merged_detections_across_years.gpkg')
    last_metric_file = os.path.join(output_dir, 'reliability_merged_across_years_single_class.jpeg')
    if os.path.exists(last_written_file) and (not assess or os.path.exists(last_metric_file)) and not OVERWRITE:           
        logger.success(f"{DONE_MSG} All files already exist in folder {output_dir}. Exiting.")
        sys.exit(0)

    all_years_dets_gdf['merged_id'] = all_years_dets_gdf.index
    encoded_geoms = all_years_dets_gdf.geometry.to_wkb()
    all_years_dets_gdf['wkb_geom'] = encoded_geoms.apply(lambda x: md5(x))

    logger.info('Merge overlapping components while ignoring the year...')
    dets_to_merge_gdf = all_years_dets_gdf.drop(columns='group_id')
    dets_to_merge_gdf.loc[:, 'geometry'] = dets_to_merge_gdf.buffer(5)
    intersecting_detections_gdf = group_detections(dets_to_merge_gdf, 0.5, ignore_year=True)
    intersecting_detections_gdf.sort_values('merged_score', ascending=False, inplace=True)
    intersecting_detections_gdf.drop(columns='wkb_geom', inplace=True)

    merged_dets_across_years_gdf = intersecting_detections_gdf.dissolve('group_id', aggfunc='first', as_index=False)
    min_year_df = intersecting_detections_gdf[['group_id', 'year_det']].groupby('group_id', as_index=False).min().rename(columns={'year_det': 'first_year'})
    max_year_df = intersecting_detections_gdf[['group_id', 'year_det']].groupby('group_id', as_index=False).max().rename(columns={'year_det': 'last_year'})
    count_year_df = intersecting_detections_gdf.group_id.value_counts().reset_index().rename(columns={'count': 'count_years'})

    for df in  [min_year_df, max_year_df, count_year_df]:
        merged_dets_across_years_gdf = pd.merge(merged_dets_across_years_gdf, df, on='group_id', how='left')
    # merged_dets_across_years_gdf.loc[:, 'merged_score'] = merged_dets_across_years_gdf.apply(
    #     lambda x: min(1, x['merged_score'] + 0.1 * (x['count_years']-1)), axis=1
    # )

    merged_dets_across_years_gdf.loc[:, 'geometry'] = merged_dets_across_years_gdf.buffer(-5)
    merged_dets_across_years_gdf.set_crs(2056, inplace=True)

    merged_dets_across_years_gdf.to_file(last_written_file, crs='EPSG:2056', index=False)
    written_files.append(last_written_file)

    logger.success(f"{DONE_MSG} {len(merged_dets_across_years_gdf)} features were left after merging across years.")

    if assess:
        written_files.extend(
            perform_assessment(
                merged_dets_across_years_gdf, labels_path, categories_path, method, output_dir,
                score='merged_score', additional_columns=['score', 'first_year', 'last_year', 'count_years'], drop_year=True,
                tagged_results_filename='tagged_merged_results_across_years', reliability_diagram_filename='reliability_diagram_merged_results_across_years'
            )
        )


if __name__ == "__main__":

    # Chronometer
    tic = time.time()
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description="The script post-process the detections to merge overlapping ones across years.")
    parser.add_argument('config_file', type=str, help='input geojson path')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")

    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    WORKING_DIR = cfg['working_directory']
    OUTPUT_DIR = cfg['output_directory']

    DETECTIONS = cfg['detections']

    written_files = []
    os.chdir(WORKING_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info('Read files...')
    detections_gdf = gpd.read_file(DETECTIONS)

    ASSESS = cfg['assess']['enable']
    if ASSESS:
        METHOD = cfg['assess']['metrics_method']
        LABELS = cfg['labels'] if 'labels' in cfg.keys() else None
        CATEGORIES = cfg['categories']
    else:
        METHOD, LABELS, CATEGORIES = (None, None, None)

    written_files = main(detections_gdf, ASSESS, METHOD, LABELS, CATEGORIES, OUTPUT_DIR)

    # Stop chronometer  
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(time.time()-tic):.2f} seconds")