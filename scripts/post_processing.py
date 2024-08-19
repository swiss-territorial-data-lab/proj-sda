import argparse
import os
import sys
import time
import yaml

import geopandas as gpd
import pandas as pd
import rasterio

sys.path.insert(0, '.')
from helpers import misc
from helpers.constants import DONE_MSG

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
    DETECTIONS = cfg['detections']
    TILES = cfg['aoi_tiles']
    TICINO_CAD = cfg['ticino']['cadaster']
    DEM = cfg['dem']
    SCORE = cfg['score']
    AREA = cfg['area']
    ELEVATION = cfg['elevation']
    DISTANCE = cfg['distance']

    written_files = [] 

    # Convert input detections to a geodataframe 
    detections_gdf = gpd.read_file(DETECTIONS)
    detections_gdf = detections_gdf.to_crs(2056)
    detections_gdf = detections_gdf[detections_gdf['tag']!='FN']
    detections_gdf['area'] = detections_gdf.geometry.area 
    detections_gdf['det_id'] = detections_gdf.index
    total = len(detections_gdf)
    logger.info(f"{total} input shapes")

    tiles_gdf = gpd.read_file(TILES)
    tiles_gdf = tiles_gdf.to_crs(2056)

    ticino_cad_gdf = gpd.read_file(TICINO_CAD)
    ticino_cad_gdf = ticino_cad_gdf.to_crs(2056)

    # Discard polygons detected at/below 0 m and above the threshold elevation
    r = rasterio.open(DEM)
    detections = detections_gdf.loc[detections_gdf['geometry'].is_valid, :] 
    row, col = r.index(detections.centroid.x, detections.centroid.y)
    values = r.read(1)[row, col]
    detections['elevation'] = values 
    detections = detections[(detections.elevation != 0) & (detections.elevation < ELEVATION)]
    te = len(detections)
    logger.info(f"{total - te} detections were removed by elevation threshold: {ELEVATION} m")

    # Merge close features
    detections_year = gpd.GeoDataFrame()

    for year in detections.year_det.unique():
    # for year in ['2004'] :
        detections_gdf = detections.copy()
        detections_temp_gdf = detections_gdf[detections_gdf['year_det']==year]

        detections_temp_buffer_gdf = detections_temp_gdf.copy()
        detections_temp_buffer_gdf['geometry'] = detections_temp_gdf.geometry.buffer(DISTANCE, resolution=2)

        detections_tiles_join_gdf = gpd.sjoin(tiles_gdf, detections_temp_buffer_gdf, how='left', predicate='contains')
        # detections_tiles_join_gdf.to_file(f'{DETECTIONS[:-5]}_test.gpkg')
     
        remove_det_list = []

        for tile_id in detections_tiles_join_gdf.id.unique():

            detection_tiles = detections_tiles_join_gdf.copy()
            detection_tiles_temp = detection_tiles[detection_tiles['id']==tile_id]  
            remove_det_list.extend(detection_tiles_temp.loc[:, 'det_id'].tolist()) 

        detections_temp2_gdf = detections_temp_gdf[~detections_temp_gdf.det_id.isin(remove_det_list)].drop_duplicates(subset=['det_id'], ignore_index=True)
        detections_temp3_gdf = detections_temp_gdf[detections_temp_gdf.det_id.isin(remove_det_list)].drop_duplicates(subset=['det_id'], ignore_index=True)

        detections_merge = detections_temp2_gdf.buffer(DISTANCE, resolution=2).geometry.unary_union
        detections_merge = gpd.GeoDataFrame(geometry=[detections_merge], crs=detections.crs)  
        if detections_merge.isnull().values.any():
            detections_merge = gpd.GeoDataFrame()
        else:
            detections_merge = detections_merge.explode(index_parts=True).reset_index(drop=True)   
            detections_merge.geometry = detections_merge.geometry.buffer(-DISTANCE, resolution=2)

        detections_temp3_gdf = detections_temp3_gdf.drop(['score', 'tag', 'dataset', 'label_class', 'CATEGORY', 'det_class',
       'det_category', 'year_det', 'year_label', 'area', 'det_id',
       'elevation'], axis=1)

        detections_merge = pd.concat([detections_merge, detections_temp3_gdf], axis=0, ignore_index=True)
        detections_merge['index_merge'] = detections_merge.index
 
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
            ['index_merge', 'dataset', 'label_class', 'CATEGORY', 
            'year_det', 'year_label']], 
            on='index_merge')
        detections_year = pd.concat([detections_year, detections_merge])

    detections_year = detections_year.drop_duplicates(subset='score')
    td = len(detections_year)
    logger.info(f"{td} clustered detections remains after shape union (distance threshold = {DISTANCE} m)")

    # Filter dataframe by score value
    detections_score = detections_year[detections_year.score > SCORE]
    sc = len(detections_score)
    logger.info(f"{td - sc} detections were removed by score filtering (score threshold = {SCORE})")

    # Discard polygons with area under the threshold 
    detections_area = detections_score[detections_score.area > AREA]
    ta = len(detections_area)
    logger.info(f"{sc - ta} detections were removed by area filtering (area threshold = {AREA} m2)")

    # # Discard polygons intersecting relevant layers

    detections_area['det_id'] = detections_area.index
    detections_area['geometry'] = detections_area.geometry.buffer(-8)
    ticino_cad_gdf['cad_id'] = ticino_cad_gdf.index

    detections_join = gpd.sjoin(detections_area, ticino_cad_gdf, how='left', predicate='intersects')
    detections_filtered = detections_join[detections_join.cad_id.isnull()].copy()
    detections_filtered['geometry'] = detections_filtered.geometry.buffer(8)

    # Final gdf
    logger.info(f"{len(detections_filtered)} detections remaining after filtering")

    # Formatting the output name of the filtered detection  
    feature = f'{DETECTIONS[:-5]}_threshold_score-{SCORE}_area-{int(AREA)}_elevation-{int(ELEVATION)}_distance-{int(DISTANCE)}'.replace('0.', '0dot') + '.gpkg'
    detections_filtered.to_file(feature)

    written_files.append(feature)
    logger.success(f"{DONE_MSG} A file was written: {feature}")  

    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()