import argparse
import os
import sys
import time
import yaml

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
from helpers import misc
from helpers.constants import DONE_MSG

from loguru import logger
logger = misc.format_logger(logger)


def plot_barchart(dir_plots, df, cat, data):

    fig, ax = plt.subplots(1, 1, figsize=(30,5))

    if data == 'label':
        df = df[df['CATEGORY']==cat].copy() 
        df = df[~(df.tag.isin(["FP", "wrong class", "small polygon"]))]
        year = 'year_label'
    elif data == 'det':
        df = df[df['det_category']==cat].copy() 
        df = df[~(df.tag.isin(["FN", "wrong class", "small polygon"]))]
        year = 'year_det'
  
    df = df[[year, 'tag']]

    df['counts'] = 1

    df = pd.pivot_table(data=df, index=[year], columns=['tag'], values='counts', aggfunc='count')

    if data == 'label':  
        
        df = df[['TP', 'FN']] 
        ax = df.plot.bar(rot=0, log=False, stacked=True, color=['turquoise', 'gold'] , width=0.8)
    elif data == 'det':    

        if 'FP' not in df.keys(): 
            df = df[['TP']] 
        else:
            df = df[['TP', 'FP']] 
        ax = df.plot.bar(rot=0, log=False, stacked=True, color=['turquoise', 'red'] , width=0.8)

    for c in ax.containers:
        labels = [int(a) if a > 0 else "" for a in c.datavalues]
        ax.bar_label(c, label_type='center', color="black", labels=labels, fontsize=8)

    plt.gca().set_yticks(plt.gca().get_yticks().tolist())

    plt.xticks(rotation=45, fontsize=8, ha='center')
    plt.xlabel(year.replace("_", " "), fontweight='bold')

    plt.title(cat, fontweight='bold')
    plt.legend(loc='upper left', frameon=False)    

    plot_path = os.path.join(dir_plots, f'{data}_{cat}.png')  
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)


    return plot_path


if __name__ == "__main__":

    # Chronometer
    tic = time.time()
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description="The script provide some plots to analyse the results")
    parser.add_argument('config_file', type=str, help='input geojson path')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")

    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    DETECTIONS = cfg['detections']
    OUTPUT = cfg['output_dir']

    written_files = [] 

    # Convert input detections to a geodataframe 
    detections_gdf = gpd.read_file(DETECTIONS)
    detections_gdf = detections_gdf.to_crs(2056)
    detections_gdf['area'] = detections_gdf.geometry.area 
    total = len(detections_gdf)
    logger.info(f"{total} input shapes")

    for cat in filter(None, detections_gdf.CATEGORY.unique()): 
        feature = plot_barchart(OUTPUT, detections_gdf, cat, data='label')
        written_files.append(feature)
        logger.success(f"{DONE_MSG} A file was written: {feature}")  
        feature = plot_barchart(OUTPUT, detections_gdf, cat, data='det')
        written_files.append(feature)
        logger.success(f"{DONE_MSG} A file was written: {feature}")  
    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()