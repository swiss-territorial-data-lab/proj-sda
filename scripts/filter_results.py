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

    gdf['overlap'] = gdf.apply(lambda row: misc.overlap(row['geometry'], row[f'{key}_geom']) if row['geometry'] and row[f'{key}_geom'] != None else 0, axis=1)
    # print(gdf['overlap'])
    # print()
    # exit(gdf['overlap'])
    # geom1 = gdf.geometry.values.tolist()
    # print('*****')
    # print(key)
    # print(len(gdf[f'{key}_geom']))
    # geom2 = gdf[f'{key}_geom'].values.tolist() 
    # print('-----')   
    # overlap = []
    # for n in range(len(gdf)):
    #     i = geom1[n] 
    #     ii = geom2[n] 
    #     if i == None or ii == None:
    #         overlap.append(0)
    #     else:
    #         overlap.append(misc.overlap(i, ii))
    # gdf['overlap'] = overlap
    # # print(gdf['overlap'])
    # print('hello')

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
    AOI = cfg['aoi']
    DETECTIONS = cfg['detections']
    CANTON = cfg['canton'][0].lower() + cfg['canton'][1:]
    AGRI_AREA = cfg['zones_infos'][CANTON]['agri_area'] if 'agri_area' in cfg['zones_infos'][CANTON].keys() else None
    BAT = cfg['zones_infos'][CANTON]['batiments'] if 'batiments' in cfg['zones_infos'][CANTON].keys() else None
    BAT_BATIR = cfg['zones_infos'][CANTON]['batiments_batir'] if 'batiments_batir' in cfg['zones_infos'][CANTON].keys() else None
    FORESTS = cfg['zones_infos'][CANTON]['forests'] if 'forests' in cfg['zones_infos'][CANTON].keys() else None
    GRANDS_COURS_EAU = cfg['zones_infos'][CANTON]['grands_cours_eau'] if 'grands_cours_eau' in cfg['zones_infos'][CANTON].keys() else None
    LANDING_STRIP = cfg['zones_infos'][CANTON]['landing_strip'] if 'landing_strip' in cfg['zones_infos'][CANTON].keys() else None
    PROTECTED_AREA = cfg['zones_infos'][CANTON]['protected_area'] if 'protected_area' in cfg['zones_infos'][CANTON].keys() else None
    PROTECTED_UG_WATER = cfg['zones_infos'][CANTON]['protected_underground_water'] if 'protected_underground_water' in cfg['zones_infos'][CANTON].keys() else None
    SDA = cfg['zones_infos'][CANTON]['sda'] if 'sda' in cfg['zones_infos'][CANTON].keys() else None
    SITES_POLLUES = cfg['zones_infos'][CANTON]['sites_pollues'] if 'sites_pollues' in cfg['zones_infos'][CANTON].keys() else None
    SLOPE = cfg['zones_infos'][CANTON]['slope'] if 'slope' in cfg['zones_infos'][CANTON].keys() else None
    DEM = cfg['dem']
    SCORE_THD = cfg['score_threshold']
    AREA_THD = cfg['area_threshold']
    ELEVATION_THD = cfg['elevation_threshold']

    os.chdir(WORKING_DIR)
    logger.info(f'Working directory set to {WORKING_DIR}')

    logger.info(f'Canton: {CANTON}')

    written_files = [] 

    # Convert input detections to a geodataframe 
    detections_gdf = gpd.read_file(DETECTIONS)
    detections_gdf = detections_gdf.to_crs(2056)
    if 'tag' in detections_gdf.keys():
        detections_gdf = detections_gdf[detections_gdf['tag']!='FN']
    detections_gdf['area'] = detections_gdf.geometry.area 
    detections_gdf['det_id'] = detections_gdf.index
    total = len(detections_gdf)
    logger.info(f"{total} detections")

    aoi_gdf = gpd.read_file(AOI)
    aoi_gdf = aoi_gdf.to_crs(2056)

    if LANDING_STRIP:
        pa_gdf = gpd.read_file(LANDING_STRIP)
        pa_gdf = pa_gdf.to_crs(2056)
        pa_gdf['pistes_avion_id'] = pa_gdf.index
    else:
        pa_gdf = gpd.GeoDataFrame()
    if BAT:
        bat_gdf = gpd.read_file(BAT)
        bat_gdf = bat_gdf.to_crs(2056)
        bat_gdf['batiments_id'] = bat_gdf.index
    else:
        bat_gdf = gpd.GeoDataFrame()
    if BAT_BATIR:
        bat_bat_gdf = gpd.read_file(BAT_BATIR)
        bat_bat_gdf = bat_bat_gdf.to_crs(2056)
        bat_bat_gdf['batiments_batir_id'] = bat_bat_gdf.index
    else:
        bat_bat_gdf = gpd.GeoDataFrame()
    if SITES_POLLUES:
        sites_pollues_gdf = gpd.read_file(SITES_POLLUES)
        sites_pollues_gdf = sites_pollues_gdf.to_crs(2056)
        sites_pollues_gdf['sites_pollues_id'] = sites_pollues_gdf.index
    else:
        sites_pollues_gdf = gpd.GeoDataFrame()
    if SDA:
        sda_gdf = gpd.read_file(SDA)
        sda_gdf = sda_gdf.to_crs(2056)
        sda_gdf['sda_id'] = sda_gdf.index
    else:
        sda_gdf = gpd.GeoDataFrame()
    feature = f'./layers/{CANTON[0].upper() + CANTON[1:]}/slope.gpkg'
    if os.path.isfile(feature):
        logger.info(f'{feature} already exists.')
        slope_gdf = gpd.read_file(feature)
    else:
        logger.info(f'{feature} does not exist and will be created.')
        slope_gdf = gpd.read_file(SLOPE)
        slope_gdf = slope_gdf.to_crs(2056)
        slope_gdf = slope_gdf[slope_gdf['HL_Klasse']!='hang_18'] 
        slope_gdf = slope_gdf.dissolve()
        slope_gdf['slope_>18%_id'] = slope_gdf.index
        slope_gdf.to_file(feature)

    info_dict = {'batiments': bat_gdf, 'batiments_batir': bat_bat_gdf, 'landing_strip': pa_gdf, 'sda': sda_gdf,
    'sites_pollues': sites_pollues_gdf, 'slope_>18%': slope_gdf}


    AGRI_AREA = cfg['zones_infos'][CANTON]['agri_area'] if 'agri_area' in cfg['zones_infos'][CANTON].keys() else None
    BAT = cfg['zones_infos'][CANTON]['batiments'] if 'batiments' in cfg['zones_infos'][CANTON].keys() else None
    BAT_BATIR = cfg['zones_infos'][CANTON]['batiments_batir'] if 'batiments_batir' in cfg['zones_infos'][CANTON].keys() else None
    FORESTS = cfg['zones_infos'][CANTON]['forests'] if 'forests' in cfg['zones_infos'][CANTON].keys() else None
    GRANDS_COURS_EAU = cfg['zones_infos'][CANTON]['grands_cours_eau'] if 'grands_cours_eau' in cfg['zones_infos'][CANTON].keys() else None
    LANDING_STRIP = cfg['zones_infos'][CANTON]['landing_strip'] if 'landing_strip' in cfg['zones_infos'][CANTON].keys() else None
    PROTECTED_AREA = cfg['zones_infos'][CANTON]['protected_area'] if 'protected_area' in cfg['zones_infos'][CANTON].keys() else None
    PROTECTED_UG_WATER = cfg['zones_infos'][CANTON]['protected_underground_water'] if 'protected_underground_water' in cfg['zones_infos'][CANTON].keys() else None
    SDA = cfg['zones_infos'][CANTON]['sda'] if 'sda' in cfg['zones_infos'][CANTON].keys() else None
    SITES_POLLUES 





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

    # Filter dataframe by score value
    detections_gdf = detections_area_gdf.copy()
    detections_score_gdf = detections_gdf[detections_gdf.score > SCORE_THD]
    sc = len(detections_score_gdf)
    logger.info(f"{ta - sc} detections were removed by score filtering (score threshold = {SCORE_THD})")

    # Indicate if polygons are intersecting relevant vector layers
    detections_infos_gdf = detections_score_gdf.copy()
    for key in info_dict.keys():
        print(key)
        if info_dict[key].empty:
            pass
        else:
            gdf = info_dict[key].copy()
            gdf = gpd.clip(gdf, aoi_gdf)
            gdf[f'{key}_geom'] = gdf.geometry 
            detections_temp_gdf = detections_score_gdf.copy() 
            detections_join_gdf = gpd.sjoin(detections_temp_gdf, gdf, how='left', predicate='intersects')      
            detections_infos_gdf = pd.merge(detections_infos_gdf, detections_join_gdf[[f'{key}_geom', 'det_id']], on='det_id', how='left')
            detections_infos_gdf[f'{key}'] = detections_infos_gdf.apply(lambda x: misc.overlap(x['geometry'], x[f'{key}_geom']) if x['geometry'] and x[f'{key}_geom'] != None else 0, axis=1)
            detections_infos_gdf = detections_infos_gdf.drop(columns=[f'{key}_geom'])
            detections_infos_gdf = detections_infos_gdf.groupby('det_id',sort=False).apply(lambda x: x if len(x)==1 else x.loc[x[f'{key}'].ne('no')]).reset_index(drop=True)
    # Compute the nearest distance between detections and sda
    detections_infos_gdf = gpd.sjoin_nearest(detections_infos_gdf, sda_gdf[['sda_id', 'geometry']], how='left', distance_col='distance_sda')
    detections_infos_gdf = detections_infos_gdf.drop_duplicates(subset=['det_id', 'sda'])
    detections_infos_gdf = detections_infos_gdf.drop(columns=['index_right', 'sda_id']) 

    # Final gdf
    logger.info(f"{len(detections_infos_gdf)} detections remaining after filtering")

    # Formatting the output name of the filtered detection  
    feature = f'{DETECTIONS[:-5]}_threshold_score-{SCORE_THD}_area-{int(AREA_THD)}_elevation-{int(ELEVATION_THD)}'.replace('0.', '0dot') + '.gpkg'
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