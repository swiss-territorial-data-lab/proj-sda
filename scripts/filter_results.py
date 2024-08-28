import argparse
import os
import sys
import time
import yaml

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, '.')
import functions.fct_misc as misc
from functions.constants import DONE_MSG

from loguru import logger
logger = misc.format_logger(logger)


def compare_geom(gdf, key):

    geom1 = gdf.geometry.values.tolist()
    geom2 = gdf[f'{key}_geom'].values.tolist()    
    overlap = []
    for (i, ii) in zip(geom1, geom2):
        if i == None or ii == None:
            overlap.append(0)
        else:
            overlap.append(misc.overlap(i, ii))
    gdf['overlap'] = overlap

    return gdf


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
    WORKING_DIR = cfg['working_dir']
    DETECTIONS = cfg['detections']
    PISTES_AVION_VD = cfg['zones_exclues']['vaud']['pistes_avion']
    BAT_VD = cfg['zones_exclues']['vaud']['batiments']
    BAT_BATIR_TC = cfg['zones_exclues']['ticino']['batiments_batir']
    SITES_POLLUES_VD = cfg['zones_infos']['vaud']['sites_pollues']
    SDA_VD = cfg['zones_infos']['vaud']['sda']
    SDA_TC = cfg['zones_infos']['ticino']['sda']
    SLOPE_VD = cfg['zones_infos']['vaud']['slope']
    SLOPE_TC = cfg['zones_infos']['ticino']['slope']
    DEM = cfg['dem']
    SCORE_THD = cfg['score_threshold']
    AREA_THD = cfg['area_threshold']
    ELEVATION_THD = cfg['elevation_threshold']
    OVERLAP_THD = cfg['overlap_threshold']

    os.chdir(WORKING_DIR)
    logger.info(f'Working directory set to {WORKING_DIR}')

    written_files = [] 

    # Convert input detections to a geodataframe 
    detections_gdf = gpd.read_file(DETECTIONS)
    detections_gdf = detections_gdf.to_crs(2056)
    detections_gdf = detections_gdf[detections_gdf['tag']!='FN']
    detections_gdf['area'] = detections_gdf.geometry.area 
    detections_gdf['det_id'] = detections_gdf.index
    total = len(detections_gdf)
    logger.info(f"{total} detections")

    pa_vd_gdf = gpd.read_file(PISTES_AVION_VD)
    pa_vd_gdf = pa_vd_gdf.to_crs(2056)
    pa_vd_gdf['pa_vd_id'] = pa_vd_gdf.index
    bat_vd_gdf = gpd.read_file(BAT_VD)
    bat_vd_gdf = bat_vd_gdf.to_crs(2056)
    bat_vd_gdf['bat_vd_id'] = bat_vd_gdf.index
    bat_tc_gdf = gpd.read_file(BAT_BATIR_TC)
    bat_tc_gdf = bat_tc_gdf.to_crs(2056)
    bat_tc_gdf['bat_tc_id'] = bat_tc_gdf.index
    sites_pollues_vd_gdf = gpd.read_file(SITES_POLLUES_VD)
    sites_pollues_vd_gdf = sites_pollues_vd_gdf.to_crs(2056)
    sites_pollues_vd_gdf['sites_pollues_vd_id'] = sites_pollues_vd_gdf.index
    sda_vd_gdf = gpd.read_file(SDA_VD)
    sda_vd_gdf = sda_vd_gdf.to_crs(2056)
    sda_tc_gdf = gpd.read_file(SDA_TC)
    sda_tc_gdf = sda_tc_gdf.to_crs(2056)
    sda_gdf = pd.concat([sda_vd_gdf, sda_tc_gdf], axis=0)
    sda_gdf['sda_id'] = sda_gdf.index
    feature = './layers/slope.gpkg'
    if os.path.isfile(feature):
        logger.info(f'{feature} already exists.')
        slope_gdf = gpd.read_file(feature)
    else:
        logger.info(f'{feature} does not exist and will be created.')
        slope_vd_gdf = gpd.read_file(SLOPE_VD)
        slope_vd_gdf = slope_vd_gdf.to_crs(2056)
        slope_vd_gdf = slope_vd_gdf[slope_vd_gdf['HL_Klasse']!='hang_18'] 
        slope_tc_gdf = gpd.read_file(SLOPE_TC)
        slope_tc_gdf = slope_tc_gdf.to_crs(2056)
        slope_tc_gdf = slope_tc_gdf[slope_tc_gdf['HL_Klasse']!='hang_18'] 
        slope_gdf = pd.concat([slope_vd_gdf, slope_tc_gdf], axis=0)
        slope_gdf = slope_gdf.dissolve()
        slope_gdf['slope_>18%_id'] = slope_gdf.index
        slope_gdf.to_file(feature)

    exclude_dict = {'piste_avion_vd': pa_vd_gdf, 'batiment_batir_vd': bat_vd_gdf, 'batiment_batir_tc': bat_tc_gdf} 
    # info_dict = {'sites_pollues_vd': sites_pollues_vd_gdf, 'sda': sda_gdf}
    info_dict = {'slope_>18%': slope_gdf, 'sites_pollues_vd': sites_pollues_vd_gdf, 'sda': sda_gdf}

    # Discard polygons detected at/below 0 m and above the threshold elevation and above a given slope
    dem = rasterio.open(DEM)

    detections_gdf = detections_gdf.loc[detections_gdf['geometry'].is_valid, :] 
    row, col = dem.index(detections_gdf.centroid.x, detections_gdf.centroid.y)
    elevation = dem.read(1)[row, col]
    detections_gdf['elevation'] = elevation 

    detections_gdf = detections_gdf[(detections_gdf.elevation != 0) & (detections_gdf.elevation < ELEVATION_THD)]
    tdem = len(detections_gdf)
    logger.info(f"{total - tdem} detections were removed by elevation threshold: {ELEVATION_THD} m")

    # Discard polygons with area under a given threshold 
    detections_area_gdf = detections_gdf[detections_gdf.area > AREA_THD]
    ta = len(detections_area_gdf)
    logger.info(f"{tdem - ta} detections were removed by area filtering (area threshold = {AREA_THD} m2)")

    # Remove polygons intersecting relevant vector layers with a min thd of 20% of the detection covered
    detections_area_gdf['det_id'] = detections_area_gdf.index

    for key in exclude_dict.keys():
        gdf = exclude_dict[key] 
        gdf[f'{key}_id'] = gdf.index
        gdf[f'{key}_geom'] = gdf.geometry 
        detections_join_gdf = gpd.sjoin(detections_area_gdf, gdf, how='left', predicate='intersects')
        detections_join_gdf = detections_join_gdf[detections_join_gdf[f'{key}_id'].notnull()].copy()
        detections_join_gdf = detections_join_gdf.drop_duplicates(subset='det_id') 
        detections_join_gdf = compare_geom(detections_join_gdf, key)
        detections_join_gdf = detections_join_gdf.drop(columns='index_right')
        detections_gdf = detections_join_gdf[detections_join_gdf['overlap']<0.2] 
        
    # Filter dataframe by score value
    detections_gdf = detections_area_gdf.copy()
    detections_score_gdf = detections_gdf[detections_gdf.score > SCORE_THD]
    sc = len(detections_score_gdf)
    logger.info(f"{tdem - ta - sc} detections were removed by score filtering (score threshold = {SCORE_THD})")
 
    # Indicate if polygons are intersecting relevant vector layers with a min thd of 20% of the detection covered
    detections_infos_gdf = detections_score_gdf.copy()
    for key in info_dict.keys():
        gdf = info_dict[key]
        gdf[f'{key}_geom'] = gdf.geometry 

        detections_temp_gdf = detections_score_gdf.copy()   
        detections_join_gdf = gpd.sjoin(detections_temp_gdf, gdf, how='left', predicate='intersects')
        detections_join_gdf[f'{key}'] = np.where(detections_join_gdf[f'{key}_id'].notnull(), 'yes', 'no')
        detections_infos_gdf = pd.merge(detections_infos_gdf, detections_join_gdf[[f'{key}', f'{key}_geom', 'det_id']], on='det_id', how='left')
        detections_infos_gdf = compare_geom(detections_infos_gdf, key)
        detections_infos_gdf[f'{key}'] = np.where(detections_infos_gdf['overlap'] < OVERLAP_THD, 'no', 'yes')
        detections_infos_gdf = detections_infos_gdf.drop(columns=[f'{key}_geom', 'overlap'])
        # detections_infos_gdf = detections_infos_gdf.drop_duplicates(subset=['det_id'])
        detections_infos_gdf = detections_infos_gdf.groupby('det_id',sort=False).apply(lambda x: x if len(x)==1 else x.loc[x[f'{key}'].ne('no')]).reset_index(drop=True)

    # Compute the nearest distance between detections and sda
    detections_infos_gdf = gpd.sjoin_nearest(detections_infos_gdf, sda_gdf[['sda_id', 'geometry']], how='left', distance_col='distance_sda')
    detections_infos_gdf = detections_infos_gdf.drop_duplicates(subset=['det_id', 'sda'])
    detections_infos_gdf = detections_infos_gdf.drop(columns=['index_right', 'sda_id']) 

    # # Final gdf
    logger.info(f"{len(detections_infos_gdf)} detections remaining after filtering")

    # Formatting the output name of the filtered detection  
    feature = f'{DETECTIONS[:-5]}_threshold_score-{SCORE_THD}_area-{int(AREA_THD)}_elevation-{int(ELEVATION_THD)}_overlap-{int(OVERLAP_THD)}'.replace('0.', '0dot') + '.gpkg'
    detections_infos_gdf.to_file(feature)

    written_files.append(feature)
    logger.success(f"{DONE_MSG} A file was written: {feature}")  

    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()