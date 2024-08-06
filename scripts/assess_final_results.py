import argparse
import os
import sys
import time
import yaml

import geopandas as gpd
import numpy as np
import pandas as pd
import json
from sklearn.metrics import confusion_matrix

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
    OUTPUT_DIR = cfg['output_dir']
    CATEGORY_IDS_JSON = cfg['category_ids_json']
    AREA_THRESHOLD = cfg['area_threshold'] if 'area_threshold' in cfg.keys() else None

    os.chdir(WORKING_DIR)
    logger.info(f'Working directory set to {WORKING_DIR}.')
    # let's make the output directory in case it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    written_files = [] 

    # Read shapefiles 
    labels = gpd.read_file(LABELS)
    labels = labels.to_crs(2056)
    labels.rename(columns={'name':'CATEGORY', 'id': 'label_class'},inplace=True)

    nb_labels = len(labels)

    logger.info(f'There are {nb_labels} polygons in {os.path.basename(LABELS)}')

    # labels['label_class'] = labels['Classe'].apply(lambda x: 0 if x == 'Activité non agricole' else 1)
    
    detections = gpd.read_file(DETECTIONS)
    detections = detections.to_crs(2056)
    detections = detections.drop(labels=['label_class', 'CATEGORY', 'year_label'], axis=1)
    nb_detections = len(detections)
    logger.info(f'There are {nb_detections} polygons in {os.path.basename(DETECTIONS)}')

    # get labels ids
    filepath = open(os.path.join(OUTPUT_DIR, 'category_ids.json'))
    categories_json = json.load(filepath)
    filepath.close()

    labels_gdf = labels.copy()
    labels_gdf.rename(columns={'Classe':'CATEGORY'}, inplace=True)

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

    IOU_THRESHOLD = 0.1
    metrics_dict = {}
    metrics_dict_by_cl = []
    metrics_df_dict = {}
    metrics_cl_df_dict = {}
    tagged_dets_gdf = gpd.GeoDataFrame()

    tp_gdf, fp_gdf, fn_gdf, mismatched_class_gdf, small_poly_gdf = metrics.get_fractional_sets(
        detections, labels_w_id_gdf, IOU_THRESHOLD, AREA_THRESHOLD)
    
    tp_gdf['tag'] = 'TP'
    fp_gdf['tag'] = 'FP'
    fn_gdf['tag'] = 'FN'
    mismatched_class_gdf['tag'] = 'wrong class'
    small_poly_gdf['tag'] = 'small polygon'

    print(len(tp_gdf), len(fp_gdf), len(fn_gdf), len(mismatched_class_gdf))

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

    file_to_write = os.path.join(OUTPUT_DIR, f'tagged_detections_merged.gpkg')
    tagged_dets_gdf[['geometry', 'det_id', 'score', 'tag', 'label_class', 'CATEGORY', 'year', 'year_det', 'det_class', 'det_category']]\
        .to_file(file_to_write, driver='GPKG', index=False)
    written_files.append(file_to_write)

    # Save the metrics by class for each dataset
    metrics_by_cl_df = pd.DataFrame()
    dataset_df = metrics_cl_df_dict.copy()
    metrics_by_cl_df = pd.concat([metrics_by_cl_df, dataset_df], ignore_index=True)

    metrics_by_cl_df['category'] = [
        categories_info_df.loc[categories_info_df.label_class==det_class+1, 'CATEGORY'].iloc[0] 
        for det_class in metrics_by_cl_df['class'].to_numpy()
    ] 

    file_to_write = os.path.join(OUTPUT_DIR, 'metrics_by_class.csv')
    metrics_by_cl_df[
        ['class', 'category', 'TP_k', 'FP_k', 'FN_k', 'precision_k', 'recall_k']
    ].sort_values(by=['class']).to_csv(file_to_write, index=False)
    written_files.append(file_to_write)

    # tmp_df = metrics_by_cl_df[['TP_k', 'FP_k', 'FN_k']].sum()
    # tmp_df2 =  metrics_by_cl_df[['precision_k', 'recall_k']].mean()
    # global_metrics_df = tmp_df.merge(tmp_df2)

    # file_to_write = os.path.join(OUTPUT_DIR, 'global_metrics.csv')
    # global_metrics_df.to_csv(file_to_write, index=False)
    # written_files.append(file_to_write)

    # Save the confusion matrix
    na_value_category = tagged_dets_gdf.CATEGORY.isna()
    sorted_classes =  tagged_dets_gdf.loc[~na_value_category, 'CATEGORY'].sort_values().unique().tolist() + ['background']
    tagged_dets_gdf.loc[na_value_category, 'CATEGORY'] = 'background'
    tagged_dets_gdf.loc[tagged_dets_gdf.det_class.isna(), 'det_class'] = 'background'


    tagged_gdf = tagged_dets_gdf.copy()

    true_class = tagged_gdf.CATEGORY.to_numpy()
    detected_class = tagged_gdf.det_class.to_numpy()

    # confusion_array = confusion_matrix(true_class, detected_class, labels=sorted_classes)
    # confusion_df = pd.DataFrame(confusion_array, index=sorted_classes, columns=sorted_classes, dtype='int64')
    # confusion_df.rename(columns={'background': 'missed labels'}, inplace=True)

    # file_to_write = (os.path.join(OUTPUT_DIR, 'confusion_matrix.csv'))
    # confusion_df.to_csv(file_to_write)
    # written_files.append(file_to_write)

    print()
    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)

    print()

    toc = time.time()
    logger.success(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()