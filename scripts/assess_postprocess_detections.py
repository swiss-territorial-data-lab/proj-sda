import argparse
import os
import sys
import time
import yaml

import geopandas as gpd
import numpy as np
import pandas as pd
import json

sys.path.insert(0, '.')

import functions.fct_metrics as metrics
import functions.fct_misc as misc
from helpers.constants import DONE_MSG

from loguru import logger
logger = misc.format_logger(logger)


if __name__ == "__main__":

    # Chronometer
    tic = time.time()
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description="The script assess the post-processed detections")
    parser.add_argument('config_file', type=str, help='input geojson path')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")

    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    WORKING_DIR = cfg['working_dir']
    LABELS = cfg['labels']
    DETECTIONS = cfg['detections']
    DISTANCE = cfg['distance']
    IOU_THD = cfg['iou'] if 'iou' in cfg.keys() else 0.25
    AREA_THD = cfg['area'] if 'area' in cfg.keys() else None

    os.chdir(WORKING_DIR)
    logger.info(f'Working directory set to {WORKING_DIR}')

    written_files = [] 

    logger.info("Loading split AoI tiles as a GeoPandas DataFrame...")
    tiles_gdf = gpd.read_file('split_aoi_tiles.geojson')
    tiles_gdf = tiles_gdf.to_crs(2056)
    if 'year' in tiles_gdf.keys(): 
        tiles_gdf = tiles_gdf.rename(columns={"year": "year_tile"})    
    logger.success(f"{DONE_MSG} {len(tiles_gdf)} features were found.")

    logger.info("Loading detections as a GeoPandas DataFrame...")
    detections_gdf = gpd.read_file(DETECTIONS)
    detections_gdf = detections_gdf.to_crs(2056)
    detections_gdf = detections_gdf.drop(labels=['label_class', 'CATEGORY', 'year_label'], axis=1)
    detections_gdf = detections_gdf[detections_gdf['tag']!='FN']
    detections_gdf['area'] = detections_gdf.geometry.area 
    detections_gdf['det_id'] = detections_gdf.index
    logger.success(f"{DONE_MSG} {len(detections_gdf)} features were found.")

    # Merge features
    logger.info(f"Merge adjacent polygons spread across tiles with a buffer of {DISTANCE} m...")
    detections_year = gpd.GeoDataFrame()

    # Process detection by year
    for year in detections_gdf.year_det.unique():
        detections_gdf = detections_gdf.copy()
        detections_temp_gdf = detections_gdf[detections_gdf['year_det']==year]

        detections_temp_buffer_gdf = detections_temp_gdf.copy()
        detections_temp_buffer_gdf['geometry'] = detections_temp_gdf.geometry.buffer(DISTANCE, resolution=2)

        # Save id of the polygons totally contained in the tile (no merging with adjacent tiles), to prevent merging them together is they are within the thd distance 
        detections_tiles_join_gdf = gpd.sjoin(tiles_gdf, detections_temp_buffer_gdf, how='left', predicate='contains')
     
        remove_det_list = []

        for tile_id in detections_tiles_join_gdf.id.unique():

            detection_tiles = detections_tiles_join_gdf.copy()
            detection_tiles_temp = detection_tiles[detection_tiles['id']==tile_id]  
            remove_det_list.extend(detection_tiles_temp.loc[:, 'det_id'].tolist()) 

        detections_temp2_gdf = detections_temp_gdf[~detections_temp_gdf.det_id.isin(remove_det_list)].drop_duplicates(subset=['det_id'], ignore_index=True)
        detections_temp3_gdf = detections_temp_gdf[detections_temp_gdf.det_id.isin(remove_det_list)].drop_duplicates(subset=['det_id'], ignore_index=True)

        # Merge polygons within the thd distance
        detections_merge = detections_temp2_gdf.buffer(DISTANCE, resolution=2).geometry.unary_union
        detections_merge = gpd.GeoDataFrame(geometry=[detections_merge], crs=detections_gdf.crs)  
        if detections_merge.isnull().values.any():
            detections_merge = gpd.GeoDataFrame()
        else:
            detections_merge = detections_merge.explode(index_parts=True).reset_index(drop=True)   
            detections_merge.geometry = detections_merge.geometry.buffer(-DISTANCE, resolution=2)

        detections_temp3_gdf = detections_temp3_gdf.drop(['score', 'tag', 'dataset', 'det_class',
       'det_category', 'year_det', 'area', 'det_id'], axis=1)

        # Concat polygons contained within a tile and saved previously
        detections_merge = pd.concat([detections_merge, detections_temp3_gdf], axis=0, ignore_index=True)
        detections_merge['index_merge'] = detections_merge.index

        # Spatially join merged detection with raw ones to retrieve relevant information (score, area,...)
        detections_join = gpd.sjoin(detections_merge, detections_temp_gdf, how='inner', predicate='intersects')

        det_class_all = []
        det_score_all = []

        for id in detections_merge.index_merge.unique():
            detections_temp_gdf = detections_join.copy()
            detections_temp_gdf = detections_temp_gdf[(detections_temp_gdf['index_merge']==id)]
            detections_temp_gdf = detections_temp_gdf.rename(columns={'score_left': 'score'})
            det_score_all.append(detections_temp_gdf['score'].mean())

            detections_temp_gdf = detections_temp_gdf.dissolve(by='det_class', aggfunc='sum', as_index=False)
            detections_temp_gdf['det_class'] = detections_temp_gdf.loc[detections_temp_gdf['area'] == detections_temp_gdf['area'].max(), 
                                                            'det_class'].iloc[0]    

            det_class = detections_temp_gdf['det_class'].drop_duplicates().tolist()
            det_class_all.append(det_class[0])

        detections_merge['det_class'] = det_class_all
        detections_merge['score'] = det_score_all

        detections_merge = pd.merge(detections_merge, detections_join[
            ['index_merge', 'dataset', 'year_det']], 
            on='index_merge')
        detections_year = pd.concat([detections_year, detections_merge])

    detections_merged_gdf = detections_year.drop_duplicates(subset='score')
    td = len(detections_merged_gdf)
    logger.info(f"... {td} clustered detections remains after shape union")
    

    logger.info("Loading labels as a GeoPandas DataFrame...")
    labels_gdf = gpd.read_file(LABELS)
    labels_gdf = labels_gdf.to_crs(2056)
    labels_gdf.rename(columns={'name':'CATEGORY', 'id': 'label_class'},inplace=True)
    logger.success(f"{DONE_MSG} {len(labels_gdf)} features were found.")

    logger.info("Assigned labels to dataset")
    tiles_gdf['tile_geometry'] = tiles_gdf['geometry']    
    labels_tiles_sjoined_gdf = gpd.sjoin(labels_gdf, tiles_gdf, how='inner', predicate='intersects')

    if 'year_label' in labels_gdf.keys():
        labels_tiles_sjoined_gdf = labels_tiles_sjoined_gdf[labels_tiles_sjoined_gdf.year_label == labels_tiles_sjoined_gdf.year_tile]  

    labels_tiles_sjoined_gdf = labels_tiles_sjoined_gdf.drop_duplicates(subset=['geometry'], keep='first')
    labels_tiles_sjoined_gdf.drop(columns=['tile_geometry', 'index_right'], inplace=True)
    labels_tiles_sjoined_gdf.rename(columns={'id': 'tile_id'}, inplace=True)

    tiles_gdf.drop('tile_geometry', inplace=True, axis=1)

    # get labels ids
    filepath = open(os.path.join('category_ids.json'))
    categories_json = json.load(filepath)
    filepath.close()

    labels_gdf = labels_tiles_sjoined_gdf.copy()

    # get classe ids
    id_classes = range(len(categories_json))

    # append class ids to labels
    categories_info_df = pd.DataFrame()

    for key in categories_json.keys():

        categories_tmp={sub_key: [value] for sub_key, value in categories_json[key].items()}
        categories_info_df = pd.concat([categories_info_df, pd.DataFrame(categories_tmp)], ignore_index=True)

    categories_info_df.sort_values(by=['id'], inplace=True, ignore_index=True)
    categories_info_df.drop(['supercategory'], axis=1, inplace=True)

    categories_info_df.rename(columns={'name':'CATEGORY', 'id': 'label_class'},inplace=True)
    labels_gdf = labels_gdf.astype({'CATEGORY':'str'})
    labels_w_id_gdf = labels_gdf.merge(categories_info_df, on='CATEGORY', how='left')

    logger.info('Tag detections and get metrics...')

    metrics_dict = {}
    metrics_dict_by_cl = []
    metrics_df_dict = {}
    metrics_cl_df_dict = {}
    tagged_dets_gdf = gpd.GeoDataFrame()

    tp_gdf, fp_gdf, fn_gdf, mismatched_class_gdf, small_poly_gdf = metrics.get_fractional_sets(
        detections_merged_gdf, labels_w_id_gdf, IOU_THD, AREA_THD)

    tp_gdf['tag'] = 'TP'
    fp_gdf['tag'] = 'FP'
    fn_gdf['tag'] = 'FN'
    mismatched_class_gdf['tag'] = 'wrong class'
    small_poly_gdf['tag'] = 'small polygon'

    tagged_dets_gdf = pd.concat([tagged_dets_gdf, tp_gdf, fp_gdf, fn_gdf, mismatched_class_gdf], ignore_index=True)
    tp_k, fp_k, fn_k, p_k, r_k, precision, recall, f1 = metrics.get_metrics(tp_gdf, fp_gdf, fn_gdf, mismatched_class_gdf, id_classes)
    logger.info(f'precision = {precision:.3f}, recall = {recall:.3f}, f1 = {f1:.3f}')

    tagged_dets_gdf['det_category'] = [
        categories_info_df.loc[categories_info_df.label_class==det_class+1, 'CATEGORY'].iloc[0] 
        if not np.isnan(det_class) else None
        for det_class in tagged_dets_gdf.det_class.to_numpy()
    ] 

    # label classes starting at 1 and detection classes starting at 0.
    for id_cl in id_classes:
        metrics_dict_by_cl.append({
            'class': id_cl,
            'precision_k': p_k[id_cl],
            'recall_k': r_k[id_cl],
            'TP_k' : tp_k[id_cl],
            'FP_k' : fp_k[id_cl],
            'FN_k' : fn_k[id_cl],
        })

    metrics_cl_df_dict = pd.DataFrame.from_records(metrics_dict_by_cl)

    feature = os.path.join('tagged_detections_merged.gpkg')
    tagged_dets_gdf = tagged_dets_gdf.to_crs(2056)
    tagged_dets_gdf[['geometry', 'det_id', 'score', 'tag', 'label_class', 'CATEGORY', 'year', 'year_det', 'det_class', 'det_category']]\
        .to_file(feature, driver='GPKG', index=False)
    written_files.append(feature)

    # Save the metrics by class for each dataset
    metrics_by_cl_df = pd.DataFrame()
    dataset_df = metrics_cl_df_dict.copy()
    metrics_by_cl_df = pd.concat([metrics_by_cl_df, dataset_df], ignore_index=True)

    metrics_by_cl_df['category'] = [
        categories_info_df.loc[categories_info_df.label_class==det_class+1, 'CATEGORY'].iloc[0] 
        for det_class in metrics_by_cl_df['class'].to_numpy()
    ] 

    file_to_write = os.path.join('metrics_by_class.csv')
    metrics_by_cl_df[
        ['class', 'category', 'TP_k', 'FP_k', 'FN_k', 'precision_k', 'recall_k']
    ].sort_values(by=['class']).to_csv(file_to_write, index=False)
    written_files.append(file_to_write)

    # Save the confusion matrix
    na_value_category = tagged_dets_gdf.CATEGORY.isna()
    sorted_classes =  tagged_dets_gdf.loc[~na_value_category, 'CATEGORY'].sort_values().unique().tolist() + ['background']
    tagged_dets_gdf.loc[na_value_category, 'CATEGORY'] = 'background'
    tagged_dets_gdf.loc[tagged_dets_gdf.det_class.isna(), 'det_class'] = 'background'

    tagged_gdf = tagged_dets_gdf.copy()

    true_class = tagged_gdf.CATEGORY.to_numpy()
    detected_class = tagged_gdf.det_class.to_numpy()

    print()
    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)
    print()

    toc = time.time()
    logger.success(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()