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

def add_to_infos_dict(infos_dict, layer_dir, layer_name, id_name, key):
    """
    Reads a geopandas GeoDataFrame from a file and adds it to a dictionary

    Parameters
    ----------
    infos_dict : dict
        dictionary where the GeoDataFrame will be added
    layer_dir : str
        directory where the file is located
    layer_name : str
        name of the file
    id_name : str
        name of the id column that will be added to the GeoDataFrame
    key : str
        key that will be used to add the GeoDataFrame in the dictionary

    Returns
    -------
    dict
        the dictionary with the added GeoDataFrame
    """
    gdf = gpd.read_file(os.path.join(layer_dir, layer_name))
    gdf = gdf.to_crs(2056)
    gdf[id_name] = gdf.index
    infos_dict[key] = gdf
    del gdf

    return infos_dict


def check_gdf_len(gdf):
    """Check if the GeoDataFrame is empty. If True, exit the script

    Args:
        gdf (GeoDataFrame): detection polygons
    """

    try:
        assert len(gdf) > 0
    except AssertionError:
        logger.error("No detections left in the dataframe. Exit script.")
        sys.exit(1)


def none_if_undefined(cfg, key):
    
    return cfg[key] if key in cfg.keys() else None


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
    WORKING_DIR = cfg['working_directory']
    DETECTIONS = cfg['detections']
    DEM = cfg['dem']

    CANTON = cfg['canton']
    CANTON_PARAMS = cfg['infos'][CANTON]
    LAYERS_DIR = cfg['infos']['layers_directory'].replace('{canton}', CANTON)

    AGRI_AREA = none_if_undefined(CANTON_PARAMS, 'agri_area') 
    BUILDINGS = none_if_undefined(CANTON_PARAMS, 'buildings') 
    BUILD_AREAS = none_if_undefined(CANTON_PARAMS, 'building_areas') 
    CLIM_AREAS = none_if_undefined(CANTON_PARAMS, 'climatic_areas') 
    FORESTS = none_if_undefined(CANTON_PARAMS, 'forests') 
    LARGE_RIVERS = none_if_undefined(CANTON_PARAMS, 'large_rivers') 
    PROTECTED_AREA = none_if_undefined(CANTON_PARAMS, 'protected_areas') 
    PROTECTED_UG_WATER = none_if_undefined(CANTON_PARAMS, 'protected_underground_water') 
    SDA = none_if_undefined(CANTON_PARAMS, 'sda') 
    POLLUTED_SITES = none_if_undefined(CANTON_PARAMS, 'polluted_sites') 
    SLOPE = none_if_undefined(CANTON_PARAMS, 'slope') 
    WATERS = none_if_undefined(CANTON_PARAMS, 'waters') 
    ZONE_COMPATIBLE_LPN = none_if_undefined(CANTON_PARAMS, 'zone_compatible_LPN') 
    ZONE_COMPATIBLE_LPN_EXTENSIVE = none_if_undefined(CANTON_PARAMS, 'zone_compatible_LPN_extensive') 
    ZONE_NON_COMPATIBLE_LPN = none_if_undefined(CANTON_PARAMS, 'zone_non_compatible_LPN')

    ATTRIBUTE_NAMES = cfg['attribute_names']
    SCORE_THD = cfg['score_threshold']
    AREA_THD = cfg['area_threshold']
    AREA_RATIO_THD = cfg['area_ratio_threshold']
    COMPACTNESS_THD = cfg['compactness_threshold']
    ELEVATION_THD = cfg['elevation_threshold']

    ASSESS = cfg['assess']['enable']
    if ASSESS:
        METHOD = cfg['assess']['metrics_method']
        LABELS = cfg['labels'].replace('{canton}', CANTON) if 'labels' in cfg.keys() else None
        CATEGORIES = cfg['categories']

    os.chdir(WORKING_DIR)
    logger.info(f'Working directory set to {WORKING_DIR}')

    logger.info(f'Canton: {CANTON}')

    if CANTON == 'vaud':
        AOI = 'AoI/vaud/MN95_CAD_TPR_LAD_MO_VD.shp'
        EXCLUSION = ['waters']
    elif CANTON == 'ticino':
        AOI = 'AoI/ticino/limiti_cantone_2012_MN95.shp'
        EXCLUSION = ['building_areas', 'forests', 'zone_non_compatible_LPN', 'waters']

    written_files = [] 

    logger.info('Read detections and AOI...')
    # Convert input detections to a geodataframe 
    detections_gdf = gpd.read_file(DETECTIONS)
    detections_gdf = detections_gdf.to_crs(2056)
    if 'tag' in detections_gdf.keys():
        detections_gdf = detections_gdf[detections_gdf['tag']!='FN']
    detections_gdf['original_area'] = detections_gdf.geometry.area 
    detections_gdf['det_id'] = detections_gdf.index
    total = len(detections_gdf)
    logger.info(f"{total} detections")

    logger.info(f'Area of interest: {AOI}')
    aoi_gdf = gpd.read_file(AOI)
    aoi_gdf = aoi_gdf.to_crs(2056)

    logger.info('Read objects of interests...')
    infos_dict = {}
    if AGRI_AREA:
        infos_dict = add_to_infos_dict(infos_dict, LAYERS_DIR, AGRI_AREA, 'agri_id', 'agri_area')
    if BUILDINGS:
        infos_dict = add_to_infos_dict(infos_dict, LAYERS_DIR, BUILDINGS, 'buildings_id', 'buildings')
    if BUILD_AREAS:
        infos_dict = add_to_infos_dict(infos_dict, LAYERS_DIR, BUILD_AREAS, 'building_areas_id', 'building_areas')
    if CLIM_AREAS:
        infos_dict = add_to_infos_dict(infos_dict, LAYERS_DIR, CLIM_AREAS, 'climatic_areas_id', 'climatic_areas_gdf')
    if LARGE_RIVERS:
        infos_dict = add_to_infos_dict(infos_dict, LAYERS_DIR, LARGE_RIVERS, 'large_rivers_id', 'large_rivers')
    if FORESTS:
        infos_dict = add_to_infos_dict(infos_dict, LAYERS_DIR, FORESTS, 'forests_id', 'forests')
    if PROTECTED_AREA:
        infos_dict = add_to_infos_dict(infos_dict, LAYERS_DIR, PROTECTED_AREA, 'protected_id', 'protected_area')
    if PROTECTED_UG_WATER:
        infos_dict = add_to_infos_dict(infos_dict, LAYERS_DIR, PROTECTED_UG_WATER, 'protected_water_id', 'protected_water')
    if SDA:
        sda_gdf = gpd.read_file(os.path.join(LAYERS_DIR, SDA))
        sda_gdf = sda_gdf.to_crs(2056)
        sda_gdf['sda_id'] = sda_gdf.index
        infos_dict['sda'] = sda_gdf
    else:
        sda_gdf = gpd.GeoDataFrame()
    if POLLUTED_SITES:
        infos_dict = add_to_infos_dict(infos_dict, LAYERS_DIR, POLLUTED_SITES, 'polluted_sites_id', 'polluted_sites')
        polluted_sites_gdf = gpd.GeoDataFrame()
    if WATERS:
        infos_dict = add_to_infos_dict(infos_dict, LAYERS_DIR, WATERS, 'waters_id', 'waters')
    if ZONE_COMPATIBLE_LPN:
        infos_dict = add_to_infos_dict(infos_dict, LAYERS_DIR, ZONE_COMPATIBLE_LPN, 'zone_compatible_LPN_id', 'pollutezone_compatible_LPN')
    if ZONE_COMPATIBLE_LPN_EXTENSIVE:
        infos_dict = add_to_infos_dict(infos_dict, LAYERS_DIR, ZONE_COMPATIBLE_LPN_EXTENSIVE, 'zone_compatible_LPN_extensive_id', 'zone_compatible_LPN_extensive')
    if ZONE_NON_COMPATIBLE_LPN:
        infos_dict = add_to_infos_dict(infos_dict, LAYERS_DIR, ZONE_NON_COMPATIBLE_LPN, 'zone_non_compatible_LPN_id', 'zone_non_compatible_LPN')

    feature = os.path.join(LAYERS_DIR, 'slope.gpkg')
    if os.path.isfile(feature):
        logger.info(f'{feature} already exists.')
        slope_gdf = gpd.read_file(feature)
    else:
        logger.info(f'{feature} does not exist and will be created.')
        slope_gdf = gpd.read_file(os.path.join(LAYERS_DIR, SLOPE))
        slope_gdf = slope_gdf.to_crs(2056)
        slope_gdf = slope_gdf[slope_gdf['HL_Klasse']!='hang_18'] 
        slope_gdf = slope_gdf.dissolve()
        slope_gdf['slope_>18%_id'] = slope_gdf.index
        slope_gdf.to_file(feature)

    infos_dict['slope_>18%'] = slope_gdf
    del slope_gdf

    logger.info('Control altitude...')
    # Discard polygons detected at/below 0 m and above the threshold elevation and above a given slope
    dem = rasterio.open(DEM)

    detections_gdf = misc.check_validity(detections_gdf, correct=True)

    row, col = dem.index(detections_gdf.centroid.x, detections_gdf.centroid.y)
    elevation = dem.read(1)[row, col]
    detections_gdf['elevation'] = elevation 

    check_gdf_len(detections_gdf)
    detections_gdf = detections_gdf[(detections_gdf.elevation != 0) & (detections_gdf.elevation < ELEVATION_THD)]
    tdem = len(detections_gdf)
    logger.info(f"{total - tdem} detections were removed by elevation threshold: {ELEVATION_THD} m")

    # Filter dataframe by score value
    check_gdf_len(detections_gdf)
    SCORE = 'merged_score' if 'merged_score' in detections_gdf.columns else 'score'
    detections_score_gdf = detections_gdf[detections_gdf[SCORE] > SCORE_THD]
    sc = len(detections_score_gdf)
    logger.info(f"{tdem - sc} detections were removed by score filtering (score threshold = {SCORE_THD})")

    detections_gdf = detections_score_gdf.copy()

    # Overlay detections with exclusion area polygons
    check_gdf_len(detections_gdf)
    if EXCLUSION and len(EXCLUSION) > 0:
        logger.info('Dissolve exclusion areas')
        exclu_gdf = gpd.GeoDataFrame()
        for key in EXCLUSION:
            gdf = infos_dict[key].copy()
            exclu_gdf = pd.concat([exclu_gdf, gdf], axis=0)
            
            # Remove the exclusion areas from the dictionnary 
            del infos_dict[key]
        
        exclu_gdf = exclu_gdf.dissolve()
        logger.info(f"Remove part of the detections intersecting exclusion areas.")
        bd = len(detections_gdf)
        detections_gdf = detections_gdf.overlay(exclu_gdf, how='difference', keep_geom_type=False)
        logger.info(f'f"{len(detections_gdf) - bd} detections were removed.')

        del exclu_gdf, detections_score_gdf

    # Spatial join between detections and other vector layers
    check_gdf_len(detections_gdf)
    all_detections_infos_gdf = detections_gdf.copy()
    for key in tqdm(infos_dict.keys(), desc='Compute intersection between detections and other vector layers'):
        if infos_dict[key].empty:
            pass
        else:
            gdf = infos_dict[key].copy()
            gdf = gpd.clip(gdf, aoi_gdf)
            gdf[f'{key}_geom'] = gdf.geometry 
            detections_temp_gdf = detections_gdf.copy() 
            detections_join_gdf = gpd.sjoin(detections_temp_gdf, gdf[[f'{key}_geom', 'geometry']], how='left', predicate='intersects')
            all_detections_infos_gdf = pd.merge(all_detections_infos_gdf, detections_join_gdf[[f'{key}_geom', 'det_id']], on='det_id', how='left')
            del detections_temp_gdf, detections_join_gdf, gdf

            all_detections_infos_gdf[f'{key}'] = all_detections_infos_gdf.apply(
                lambda x: misc.overlap(x['geometry'], x[f'{key}_geom']) if x['geometry'] and x[f'{key}_geom'] != None else 0, axis=1
            )
            all_detections_infos_gdf = all_detections_infos_gdf.drop(columns=[f'{key}_geom'])
            key_list = all_detections_infos_gdf.columns.values.tolist()
            all_detections_infos_gdf = all_detections_infos_gdf.groupby(by=key_list[:-1], as_index=False, dropna=False).agg({key: ['sum']}).droplevel(1, axis=1)

        all_detections_infos_gdf = gpd.GeoDataFrame(all_detections_infos_gdf, crs=detections_gdf.crs, geometry='geometry')

    del infos_dict

    logger.info('Control the area left after spatial difference...')
    # Discard polygons with area under a given threshold 
    check_gdf_len(all_detections_infos_gdf)
    all_detections_infos_gdf['valid_area'] = all_detections_infos_gdf.area
    tsjoin = len(all_detections_infos_gdf)
    detections_infos_gdf = all_detections_infos_gdf[all_detections_infos_gdf.valid_area > AREA_THD].copy()
    ta = len(detections_infos_gdf)
    logger.info(f"{tsjoin - ta} detections were removed by area filtering (area threshold = {AREA_THD} m2)")

    detections_infos_gdf['area_ratio'] = round(detections_infos_gdf.valid_area / detections_infos_gdf.original_area, 2)
    detections_infos_gdf['compactness'] = round(4*np.pi*detections_infos_gdf.area / (detections_infos_gdf.length**2))  # Polsby–Popper test
    condition = (detections_infos_gdf.area_ratio > AREA_RATIO_THD) | (detections_infos_gdf.compactness > COMPACTNESS_THD)
    detections_infos_gdf = detections_infos_gdf[condition]
    ar = len(detections_infos_gdf)
    logger.info(f"{ta - ar} detections were removed by area ratio and compactness filtering")
    logger.info(f"(area ratio threshold = {AREA_RATIO_THD} & compactness threshold = {COMPACTNESS_THD})")
    check_gdf_len(detections_infos_gdf)
    del all_detections_infos_gdf

    # Compute the nearest distance between detections and sda
    if SDA:
        logger.info('Compute the nearest distance between detection polygons and SDA polygons.')
        detections_infos_gdf = gpd.sjoin_nearest(detections_infos_gdf, sda_gdf[['sda_id', 'geometry']], how='left', distance_col='distance_sda')
        detections_infos_gdf.drop_duplicates(subset=['det_id', 'sda'], inplace=True)   # Remove duplicates of dets overlapping several SDAs
        detections_infos_gdf.drop(columns=['index_right', 'sda_id'], inplace=True)

    # Final gdf
    logger.info(f"{len(detections_infos_gdf)} detections remaining after filtering")

    # Rename attribute names according to the Canton's needs
    attribute_names_df = pd.read_excel(ATTRIBUTE_NAMES, sheet_name=CANTON, engine='openpyxl')
    for i in range(len(attribute_names_df)):
        if attribute_names_df['argument'][i] in detections_infos_gdf.keys():
            detections_infos_gdf.rename(columns={attribute_names_df['argument'][i]: attribute_names_df['name'][i]}, inplace=True)

    # Formatting the output name of the filtered detection  
    feature = os.path.join(
        os.path.dirname(DETECTIONS), 
        f'dets_{SCORE}-{SCORE_THD}_A-{int(AREA_THD)}_A_ratio-{AREA_RATIO_THD}_C-{COMPACTNESS_THD}_elev-{int(ELEVATION_THD)}'.replace('0.', '0dot') + '.gpkg'
    )
    detections_infos_gdf.round(3).to_file(feature)
    written_files.append(feature)
    logger.success(f"{DONE_MSG} A file was written: {feature}")  
    logger.success(f"{DONE_MSG} {len(detections_infos_gdf)} features were kept.")
    logger.success(f'The covered area is {round(detections_infos_gdf.unary_union.area/1000000, 2)} km2.')

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

        written_files.extend(
            metrics.perform_assessment(
                detections_infos_gdf, labels_w_id_gdf, categories_info_df, METHOD, os.path.dirname(DETECTIONS),
                score=SCORE, additional_columns=['valid_area', 'area_ratio', 'compactness', 'year_label', 'year_det'],
                tagged_results_filename='tagged_final_dets', reliability_diagram_filename='final_reliability_diagram'
            )
        )

    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()