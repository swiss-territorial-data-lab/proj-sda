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


def plot_barchart(df):

    plt.rcParams["figure.figsize"] = (10, 5)
    fig, ax = plt.subplots(1, 1)

    ax = df.plot(x='year', kind='bar', stacked=True, width=0.9, color=['turquoise', 'gold'])
    plt.legend(loc='upper left', frameon=False)    

    for c in ax.containers:
        labels = [int(a) if a > 0 else "" for a in c.datavalues]
        ax.bar_label(c, label_type='center', color="black", labels=labels, fontsize=8)

    ticks_to_use = df.index[::5]
    labels = df.year[::5]

    ax.set_xticks(ticks_to_use, labels, rotation=45, fontsize=10, ha='center')
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    plt.xlabel('Year', fontweight='bold')

    plot_path = f'gt_{x}.png' 
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
    WORKING_DIR = cfg['working_dir']
    LABELS = cfg['labels']
    MIN_YEAR = cfg['min_year']
    MAX_YEAR = cfg['max_year']

    os.chdir(WORKING_DIR)

    written_files = [] 

    # Convert input labels to a geodataframe 
    labels_gdf = gpd.read_file(LABELS)
    labels_gdf = labels_gdf.to_crs(2056)

    total = len(labels_gdf)
    logger.info(f"{total} input shapes")

    x = 'year'
    y = 'Classe'
    z = 'count'

    labels_gdf[x] = labels_gdf[x].astype(int)

    year_all_list = np.arange(MIN_YEAR, MAX_YEAR, 1, dtype=int)
    df = pd.DataFrame({x: year_all_list}).sort_values(by=x).reset_index(drop=True)

    for v in labels_gdf[y].unique():
        df_temp = labels_gdf[labels_gdf[y]==v] 
        df_temp = df_temp.groupby(by=x, as_index=False)[y].value_counts()
        df_temp = df_temp.drop(columns=[y]).rename(columns={z:v})
        df = df.merge(df_temp, how='left', on=x).fillna(0)

    feature = plot_barchart(df)
    written_files.append(feature)
    logger.success(f"{DONE_MSG} A file was written: {feature}") 

    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()