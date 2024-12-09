import argparse
import os
import sys
import time
import yaml

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
import functions.fct_misc as misc
from functions.constants import DONE_MSG

from loguru import logger
logger = misc.format_logger(logger)


def plot_barchart(df):

    fig, ax = plt.subplots(1, 1, figsize=(30,5))

    ax = df.groupby('Classe')['canton'].value_counts().unstack().plot.bar(stacked=True, color=['turquoise', 'gold', 'pink'])
    plt.legend(loc='upper left', frameon=False)    

    for c in ax.containers:
        labels = [int(a) if a > 0 else "" for a in c.datavalues]
        ax.bar_label(c, label_type='center', color="black", labels=labels, fontsize=8)

    plt.xticks(rotation=0, fontsize=10, ha='center')
    plt.xlabel("")

    plot_path = 'gt.png' 
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

    os.chdir(WORKING_DIR)

    written_files = [] 

    # Convert input labels to a geodataframe 
    labels_gdf = gpd.read_file(LABELS)
    labels_gdf = labels_gdf.to_crs(2056)

    total = len(labels_gdf)
    logger.info(f"{total} input shapes")

    feature = plot_barchart(labels_gdf[['Classe', 'canton']] )
    written_files.append(feature)
    logger.success(f"{DONE_MSG} A file was written: {feature}") 

    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()