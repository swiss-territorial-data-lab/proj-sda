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
    AGRI_AREA = cfg['infos'][CANTON]['agri_area'] if 'agri_area' in cfg['infos'][CANTON].keys() else None
    BUILDINGS = cfg['infos'][CANTON]['buildings'] if 'buildings' in cfg['infos'][CANTON].keys() else None
    BUILD_AREAS = cfg['infos'][CANTON]['building_areas'] if 'building_areas' in cfg['infos'][CANTON].keys() else None
    FORESTS = cfg['infos'][CANTON]['forests'] if 'forests' in cfg['infos'][CANTON].keys() else None
    LARGE_RIVERS = cfg['infos'][CANTON]['large_rivers'] if 'large_rivers' in cfg['infos'][CANTON].keys() else None
    LANDING_STRIP = cfg['infos'][CANTON]['landing_strip'] if 'landing_strip' in cfg['infos'][CANTON].keys() else None
    PROTECTED_AREA = cfg['infos'][CANTON]['protected_area'] if 'protected_area' in cfg['infos'][CANTON].keys() else None
    PROTECTED_UG_WATER = cfg['infos'][CANTON]['protected_underground_water'] if 'protected_underground_water' in cfg['infos'][CANTON].keys() else None
    SDA = cfg['infos'][CANTON]['sda'] if 'sda' in cfg['infos'][CANTON].keys() else None
    POLLUTED_SITES = cfg['infos'][CANTON]['polluted_sites'] if 'polluted_sites' in cfg['infos'][CANTON].keys() else None
    SLOPE = cfg['infos'][CANTON]['slope'] if 'slope' in cfg['infos'][CANTON].keys() else None
    ZONE_COMPATIBLE_LPN = cfg['infos'][CANTON]['zone_compatible_LPN'] if 'zone_compatible_LPN' in cfg['infos'][CANTON].keys() else None
    ZONE_COMPATIBLE_LPN_EXTENSIVE = cfg['infos'][CANTON]['zone_compatible_LPN_extensive'] if 'zone_compatible_LPN_extensive' in cfg['infos'][CANTON].keys() else None
    ZONE_NON_COMPATIBLE_LPN = cfg['infos'][CANTON]['zone_non_compatible_LPN'] if 'zone_non_compatible_LPN' in cfg['infos'][CANTON].keys() else None
    EXCLUSION = cfg['exclusion'] if 'exclusion' in cfg.keys() else None
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

    if AGRI_AREA:
        agri_gdf = gpd.read_file(AGRI_AREA)
        agri_gdf = agri_gdf.to_crs(2056)
        agri_gdf['agri_id'] = agri_gdf.index
    else:
        agri_gdf = gpd.GeoDataFrame()
    if BUILDINGS:
        building_gdf = gpd.read_file(BUILDINGS)
        building_gdf = building_gdf.to_crs(2056)
        building_gdf['buildings_id'] = building_gdf.index
    else:
        building_gdf = gpd.GeoDataFrame()
    if BUILD_AREAS:
        building_areas = gpd.read_file(BUILD_AREAS)
        building_areas = building_areas.to_crs(2056)
        building_areas['building_areas_id'] = building_areas.index
    else:
        building_areas = gpd.GeoDataFrame()
    if LARGE_RIVERS:
        large_rivers_gdf = gpd.read_file(LARGE_RIVERS)
        large_rivers_gdf = large_rivers_gdf.to_crs(2056)
        large_rivers_gdf['large_rivers_id'] = large_rivers_gdf.index
    else:
        large_rivers_gdf = gpd.GeoDataFrame()
    if FORESTS:
        forests_gdf = gpd.read_file(FORESTS)
        forests_gdf = forests_gdf.to_crs(2056)
        forests_gdf['forests_id'] = forests_gdf.index
    else:
        forests_gdf = gpd.GeoDataFrame() 
    if LANDING_STRIP:
        ls_gdf = gpd.read_file(LANDING_STRIP)
        ls_gdf = ls_gdf.to_crs(2056)
        ls_gdf['pistes_avion_id'] = ls_gdf.index
    else:
        ls_gdf = gpd.GeoDataFrame()
    if PROTECTED_AREA:
        protected_gdf = gpd.read_file(PROTECTED_AREA)
        protected_gdf = protected_gdf.to_crs(2056)
        protected_gdf['protected_id'] = protected_gdf.index
    else:
        protected_gdf = gpd.GeoDataFrame()
    if PROTECTED_UG_WATER:
        protected_water_gdf = gpd.read_file(PROTECTED_UG_WATER)
        protected_water_gdf = protected_water_gdf.to_crs(2056)
        protected_water_gdf['protected_water_id'] = protected_water_gdf.index
    else:
        protected_water_gdf = gpd.GeoDataFrame()
    if SDA:
        sda_gdf = gpd.read_file(SDA)
        sda_gdf = sda_gdf.to_crs(2056)
        sda_gdf['sda_id'] = sda_gdf.index
    else:
        sda_gdf = gpd.GeoDataFrame()
    if POLLUTED_SITES:
        polluted_sites_gdf = gpd.read_file(POLLUTED_SITES)
        polluted_sites_gdf = polluted_sites_gdf.to_crs(2056)
        polluted_sites_gdf['polluted_sites_id'] = polluted_sites_gdf.index
    else:
        polluted_sites_gdf = gpd.GeoDataFrame()
    if ZONE_COMPATIBLE_LPN:
        zone_compatible_lpn_gdf = gpd.read_file(ZONE_COMPATIBLE_LPN)
        zone_compatible_lpn_gdf = zone_compatible_lpn_gdf.to_crs(2056)
        zone_compatible_lpn_gdf['zone_compatible_LPN_id'] = zone_compatible_lpn_gdf.index
    else:
        zone_compatible_lpn_gdf = gpd.GeoDataFrame()
    if ZONE_COMPATIBLE_LPN_EXTENSIVE:
        zone_compatible_lpn_extensive_gdf = gpd.read_file(ZONE_COMPATIBLE_LPN_EXTENSIVE)
        zone_compatible_lpn_extensive_gdf = zone_compatible_lpn_extensive_gdf.to_crs(2056)
        zone_compatible_lpn_extensive_gdf['zone_compatible_LPN_extensive_id'] = zone_compatible_lpn_extensive_gdf.index
    else:
        zone_compatible_lpn_extensive_gdf = gpd.GeoDataFrame()
    if ZONE_NON_COMPATIBLE_LPN:
        zone_non_compatible_lpn_gdf = gpd.read_file(ZONE_NON_COMPATIBLE_LPN)
        zone_non_compatible_lpn_gdf = zone_non_compatible_lpn_gdf.to_crs(2056)
        zone_non_compatible_lpn_gdf['zone_non_compatible_LPN_id'] = zone_non_compatible_lpn_gdf.index
    else:
        zone_non_compatible_lpn_gdf = gpd.GeoDataFrame()

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

    infos_dict = {'slope_>18%': slope_gdf, 'agri_area': agri_gdf, 'buildings': building_gdf, 'building_areas': building_areas, 'forests': forests_gdf,
    'large_rivers': large_rivers_gdf, 'landing_strip': ls_gdf, 'protected_area': protected_gdf, 'protected_water': protected_water_gdf, 
    'sda': sda_gdf, 'polluted_sites': polluted_sites_gdf, 'zone_compatible_LPN': zone_compatible_lpn_gdf, 
    'zone_compatible_LPN_extensive': zone_compatible_lpn_extensive_gdf, 'zone_non_compatible_LPN': zone_non_compatible_lpn_gdf}

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

    detections_gdf = detections_score_gdf.copy()

    # Overlay detections with exclusion area polygons
    if EXCLUSION and len(EXCLUSION) > 0:
        logger.info(f"Remove part of the detections intersecting exclusion areas.")
        exclu_gdf = gpd.GeoDataFrame()
        for key in EXCLUSION:
            gdf = infos_dict[key].copy()
            exclu_gdf = pd.concat([exclu_gdf, gdf], axis=0)
            
            # Remove the exclusion areas from the dictionnary 
            del infos_dict[key]
        
        exclu_gdf = exclu_gdf.dissolve()
        detections_gdf = detections_gdf.overlay(exclu_gdf, how='difference')

    # Indicate if polygons are intersecting relevant vector layers
    detections_infos_gdf = detections_gdf.copy()
    for key in infos_dict.keys():
        if infos_dict[key].empty:
            pass
        else:
            gdf = infos_dict[key].copy()
            gdf = gpd.clip(gdf, aoi_gdf)
            gdf[f'{key}_geom'] = gdf.geometry 
            detections_temp_gdf = detections_gdf.copy() 
            detections_join_gdf = gpd.sjoin(detections_temp_gdf, gdf[[f'{key}_geom', 'geometry']], how='left', predicate='intersects')  
            del detections_temp_gdf
            del gdf
            detections_infos_gdf = pd.merge(detections_infos_gdf, detections_join_gdf[[f'{key}_geom', 'det_id']], on='det_id', how='left')
            del detections_join_gdf
            detections_infos_gdf[f'{key}'] = detections_infos_gdf.apply(lambda x: misc.overlap(x['geometry'], x[f'{key}_geom']) if x['geometry'] and x[f'{key}_geom'] != None else 0, axis=1)
            detections_infos_gdf = detections_infos_gdf.drop(columns=[f'{key}_geom'])
            detections_infos_gdf = detections_infos_gdf.groupby('det_id',sort=False).apply(lambda x: x if len(x)==1 else x.loc[x[f'{key}'].ne('no')]).reset_index(drop=True)
            
    # Compute the nearest distance between detections and sda
    if SDA:
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