import argparse
import os
import sys
import time
import yaml

import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, '.')
import functions.fct_misc as misc
from functions.constants import DONE_MSG

from loguru import logger
logger = misc.format_logger(logger)


def plot_barchart(df, cat, min_year, max_year, data):

    plt.rcParams["figure.figsize"] = (12, 5)
    fig, ax = plt.subplots(1, 1)

    if data == 'label':
        df = df[df['CATEGORY']==cat].copy() 
        df = df[~(df.tag.isin(["FP", "wrong class", "small polygon"]))]
        year = 'year_label'
    elif data == 'det':
        df = df[df['det_category']==cat].copy() 
        df = df[~(df.tag.isin(["FN", "wrong class", "small polygon"]))]
        year = 'year_det'
    elif data == 'both':
        df['category'] = df['CATEGORY'].mask(df['CATEGORY'].isna(), df['det_category'])
        df = df[df['category']==cat].copy() 
        df['year'] = df['year_det'].mask(df['year_det'].isna(), df['year_label'])     
        year = 'year'

    df = df[[year, 'tag']].astype({year: 'int', 'tag': 'str'})
    df['counts'] = 1

    df_temp = pd.pivot_table(data=df, index=[year], columns=['tag'], values='counts', aggfunc='count')

    year_all_list = np.arange(min_year, max_year, 1, dtype=int)
    year_filled_df = pd.DataFrame({year: year_all_list}).sort_values(by=year).reset_index(drop=True)
    df = year_filled_df.merge(df_temp, how='left', on=year).fillna(0)

    if data == 'label':        
        df = df[[year, 'TP', 'FN']] 
        ax = df.plot(x=year, kind='bar', rot=0, log=False, stacked=True, color=['limegreen', 'red'], width=0.9)
    elif data == 'det':    
        if 'FP' not in df.keys(): 
            df = df[['TP']] 
        else:
            df = df[[year, 'TP', 'FP']] 
        ax = df.plot(x=year, kind='bar', rot=0, log=False, stacked=True, color=['limegreen', 'royalblue'], width=0.9)
    elif data == 'both':
        df = df[[year, 'TP', 'FN', 'FP']]
        ax = df.plot(x=year, kind='bar', rot=0, log=False, stacked=True, color=['limegreen', 'red', 'royalblue'], width=0.9)
    
    ## Uncomment to add bar labels
    # for c in ax.containers:
    #     labels = [int(a) if a > 0 else "" for a in c.datavalues]
    #     ax.bar_label(c, label_type='center', color="black", labels=labels, fontsize=7)

    plt.gca().set_yticks(plt.gca().get_yticks().tolist())
    ticks_to_use = df.index[::5]
    labels = df[year][::5]

    ax.set_xticks(ticks_to_use, labels, rotation=45, fontsize=10, ha='center')
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    plt.xlabel('Year', fontweight='bold')

    plt.title(cat, fontweight='bold')
    plt.legend(loc='upper left', frameon=False)    

    plot_path = f'{data}_{cat}.png'
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)

    return plot_path


def plot_boxplot(df, param):

    ax = df.boxplot(column=[param], by=['tag'], grid=False)
    if param == 'area':
        ax.set_yscale('symlog')
        ax.yaxis.set_major_formatter(lambda x, p: f'{int(x):,}')
    
    ax.set_xlabel('Detection tags', fontweight='bold')
    ax.set_ylabel(param.capitalize(), fontweight='bold')
    ax.get_figure().suptitle('')
    ax.set_title('')

    plot_path = f'{param}_boxplot.png'
    plt.savefig(plot_path, bbox_inches='tight')

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
    WORKING_DIR = cfg['working_dir']
    DETECTIONS = cfg['detections']
    MIN_YEAR = cfg['min_year'] if 'min_year' in cfg.keys() else 1945
    MAX_YEAR = cfg['max_year'] if 'max_year' in cfg.keys() else 2025

    os.chdir(WORKING_DIR)

    written_files = [] 

    # Convert input detections to a geodataframe 
    detections_gdf = gpd.read_file(DETECTIONS)
    detections_gdf = detections_gdf.to_crs(2056)
    detections_gdf['area'] = detections_gdf.area 

    total = len(detections_gdf)
    logger.info(f"{total} input shapes")

    for parameter in ['area', 'score']:
        feature = plot_boxplot(detections_gdf, param=parameter)
        written_files.append(feature)
        logger.success(f"{DONE_MSG} A file was written: {feature}") 

    for cat in filter(None, detections_gdf.CATEGORY.unique()): 
        feature = plot_barchart(detections_gdf, cat, MIN_YEAR, MAX_YEAR, data='label')
        written_files.append(feature)
        feature = plot_barchart(detections_gdf, cat, MIN_YEAR, MAX_YEAR, data='det')
        written_files.append(feature) 
        feature = plot_barchart(detections_gdf, cat, MIN_YEAR, MAX_YEAR, data='both')
        written_files.append(feature)

    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(WORKING_DIR + written_file)

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()