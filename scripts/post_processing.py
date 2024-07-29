import argparse
import os
import sys
import time
import yaml

import geopandas as gpd
import pandas as pd
import rasterio
from sklearn.cluster import KMeans

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
    parser = argparse.ArgumentParser(description="The script fpost-process the detection results obtained with the Object-Detector")
    parser.add_argument('config_file', type=str, help='input geojson path')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")

    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    # YEAR = cfg['year']
    DETECTIONS = cfg['detections']
    # SHPFILE = cfg['shapefile']
    DEM = cfg['dem']
    SCORE = cfg['score']
    AREA = cfg['area']
    ELEVATION = cfg['elevation']
    DISTANCE = cfg['distance']
    # OUTPUT = cfg['output']

    written_files = [] 

    # # Convert input detection to a geo dataframe 
    # aoi = gpd.read_file(SHPFILE)
    # aoi = aoi.to_crs(epsg=2056)
    
    detections = gpd.read_file(DETECTIONS)
    detections = detections.to_crs(2056)
    detections['area'] = detections.geometry.area 
    total = len(detections)
    logger.info(f"{total} input shapes")

    # Discard polygons detected above the threshold elevalation and 0 m 
    r = rasterio.open(DEM)
    row, col = r.index(detections.centroid.x, detections.centroid.y)
    values = r.read(1)[row, col]
    detections['elevation'] = values 
    detections = detections[(detections.elevation < ELEVATION) & (detections.elevation != 0)]
    te = len(detections)
    logger.info(f"{total - te} detections were removed by elevation threshold: {ELEVATION} m")

    # # Centroid of every detection polygon
    # centroids = gpd.GeoDataFrame()
    # centroids.geometry = detections.representative_point()

    # # KMeans Unsupervised Learning
    # centroids = pd.DataFrame({'x': centroids.geometry.x, 'y': centroids.geometry.y})
    # k = int((len(detections)/3) + 1)
    # cluster = KMeans(n_clusters=k, algorithm='auto', random_state=1)
    # model = cluster.fit(centroids)
    # labels = model.predict(centroids)
    # logger.info(f"KMeans algorithm computed with k = {k}")

    # # Dissolve and aggregate (keep the max value of aggregate attributes)
    # detections['cluster'] = labels

    # detections = detections.dissolve(by='cluster', aggfunc='max')
    # total = len(detections)

    # Merge close features
    # detections_merge = gpd.GeoDataFrame()

    detections_merge = detections.buffer(DISTANCE, resolution=2).geometry.unary_union
    detections_merge = gpd.GeoDataFrame(geometry=[detections_merge], crs=detections.crs)  
    detections_merge = detections_merge.explode(index_parts=True).reset_index(drop=True)   
    detections_merge.geometry = detections_merge.geometry.buffer(-DISTANCE, resolution=2)  
    detections_merge['index_merge'] = detections_merge.index

    detections_join = gpd.sjoin(detections_merge, detections, how='inner', predicate='intersects')
    detections_temp_all = gpd.GeoDataFrame()
    det_class_all = []
    det_score_all = []
    for id in detections_merge.index_merge.unique():

        detections_temp = detections_join.copy()
        detections_temp = detections_join[(detections_join['index_merge']==id)] 
        det_score = detections_temp['score'].mean()
        det_score_all.append(det_score)

        detections_temp = detections_temp.dissolve(by='det_class', aggfunc='sum', as_index=False)
        detections_temp['det_class'] = detections_temp.loc[detections_temp['area'] == detections_temp['area'].max(), 'det_class'].iloc[0]         
        det_class = detections_temp['det_class'].drop_duplicates().tolist()
        det_class_all.append(det_class[0])

    detections_merge['det_class'] = det_class_all
    detections_merge['score'] = det_score_all

    # print(detections_temp3[[ 'index_merge', 'det_class', 'score', 'area']])
    # detections_merge = detections_join.dissolve(by='index_merge', aggfunc='max', as_index=False)
    # detections_merge = detections_join.dissolve(by='index_merge', aggfunc='mean')

    # print(detections_merge[[ 'index_merge', 'det_class', 'score', 'area']])

    td = len(detections_merge)
    logger.info(f"{td} clustered detections remains after shape union (distance threshold = {DISTANCE} m)")


    # Filter dataframe by score value
    detections_score = detections_merge[detections_merge.score > SCORE]
    sc = len(detections_score)
    logger.info(f"{td - sc} detections were removed by score filtering (score threshold = {SCORE})")

    # # Clip detection to AoI
    # detections = gpd.clip(detections, aoi)

    # Discard polygons with area under the threshold 
    detections_area = detections_score[detections_score.area > AREA]
    ta = len(detections_area)
    logger.info(f"{sc - ta} detections were removed by area filtering (area threshold = {AREA} m2)")

    # # Preparation of a geo df 
    # data = {'id': detections_merge.index,'area': detections_merge.area, 'centroid_x': detections_merge.centroid.x, 'centroid_y': detections_merge.centroid.y, 'geometry': detections_merge}
    # geo_tmp = gpd.GeoDataFrame(data, crs=detections.crs)

    # Final gdf
    # detection_filtered = detections_area.drop(['index_right', 'crs', 'dataset'], axis=1)
    detection_filtered = detections_area
    logger.info(f"{len(detection_filtered)} detections remaining after filtering")

    # # Get the averaged detection score of the merged polygons  
    # intersection = gpd.sjoin(geo_tmp, detections, how='inner')
    # intersection['id'] = intersection.index
    # score_final = intersection.groupby(['id']).mean(numeric_only=True)

    # # Formatting the final geo df 
    # data = {'id_feature': detections_merge.index,'score': score_final['score'] , 'area': detections_merge.area, 'centroid_x': detections_merge.centroid.x, 'centroid_y': detections_merge.centroid.y, 'geometry': detections_merge}
    # detections_final = gpd.GeoDataFrame(data, crs=detections.crs)
    # logger.info(f"{len(detections_final)} detections remaining after filtering")

    # Formatting the output name of the filtered detection  
    # name = os.path.basename(DETECTIONS)
    feature = f'{DETECTIONS[:-5]}_threshold_score-{SCORE}_area-{int(AREA)}_elevation-{int(ELEVATION)}_distance-{int(DISTANCE)}'.replace('0.', '0dot') + '.gpkg'
    detection_filtered.to_file(feature)

    written_files.append(feature)
    logger.success(f"{DONE_MSG} A file was written: {feature}")  

    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()