import os, sys
import argparse
import yaml
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from loguru import logger

# the following allows us to import modules from within this file's parent folder
sys.path.insert(0, '.')

logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}", level="INFO")

if __name__ == "__main__":

    # Start chronometer
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='Framework configuration file')
    args = parser.parse_args()
    logger.info(f"Using {args.config_file} as config file.")

    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    WORKING_DIR = cfg['working_dir']
    PLOTS = cfg['plots']
    OBJECT_IDs = sorted(cfg['object_id'])
    DETECTIONS_SHP = cfg['detections_shapefile']
    OUTPUT_DIR = cfg['output_dir']

    os.chdir(WORKING_DIR)
    
    # Create an output directory in case it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Load prediction 
    gdf = gpd.read_file(DETECTIONS_SHP)
    gdf = gdf.sort_values(by=['year'], ascending=False)

    for PLOT in PLOTS:
        
        # Plot the quarry area vs time 
        if PLOT == 'area-year':
            logger.info(f"Plot {PLOT}")
            fig, ax = plt.subplots(figsize=(8,5))
            for ID in OBJECT_IDs:
                x = gdf.loc[gdf["id_object"] == ID,["year"]]
                y = gdf.loc[gdf["id_object"] == ID,["area"]]
                id = ID
                ax.scatter(x, y, label=id)
                ax.plot(x, y, linestyle="dotted")
                ax.set_xlabel("Year", fontweight='bold')
                ax.set_ylabel(r"Area (m$^2$)", fontweight='bold')
                ax.ticklabel_format(axis='y', style='sci',scilimits=(0,0))
                ax.legend(title='Object ID', loc=[1.05,0.5] )
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            plot_path = os.path.join(OUTPUT_DIR, 'quarry_area-year.png')
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.show()