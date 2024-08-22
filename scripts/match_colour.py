import argparse
import os
import sys
import time
import yaml

from glob import glob
from loguru import logger
from tqdm import tqdm

import matplotlib.pyplot as plt
import rasterio as rio
from rio_hist.match import histogram_match
from skimage import exposure


# the following allows us to import modules from within this file's parent folder
sys.path.insert(1, '.')
import functions.fct_misc as misc

logger = misc.format_logger(logger)

def plot_histogram(img, ref, matched, output, img_name):

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(18, 16))

    for i, img in enumerate((img, ref, matched)):
        # axes[0, i].imshow(img)
        # axes[0, i].imshow(reference)
        # axes[0, i].imshow(matched)
        for c, c_color in enumerate(('red', 'green', 'blue')):
            img_hist, bins = exposure.histogram(img[..., c], source_range='dtype')
            axes[c + 1, i].plot(bins, img_hist / img_hist.max())
            img_cdf, bins = exposure.cumulative_distribution(img[..., c])
            axes[c + 1, i].plot(bins, img_cdf)
            axes[c + 1, 0].set_ylabel(c_color)

    axes[0, 0].set_title('Source')
    axes[0, 1].set_title('Reference')
    axes[0, 2].set_title('Matched to R, G, B')

    plt.tight_layout()
    plt.savefig(os.path.join(output, img_name + '_histo_plot.png'))


def plot_image(img, ref, matched, output, img_name):

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                        sharex=True, sharey=True)
    for aa in (ax1, ax2, ax3):
        aa.set_axis_off()

    ax1.imshow(img)
    ax1.set_title('Source')
    ax2.imshow(ref)
    ax2.set_title('Reference')
    ax3.imshow(matched)
    ax3.set_title('Matched to R, G, B')

    plt.tight_layout()
    plt.savefig(os.path.join(output, img_name + '_image_plot.png'))


if __name__ == "__main__":

    # Start chronometer
    tic = time.time()
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description="Mosaic image")
    parser.add_argument('config_file', type=str, help='Framework configuration file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")

    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    WORKING_DIR = cfg['working_dir']
    INPUT_DIR = cfg['image_dir']
    OUTPUT_DIR = cfg['output_dir']
    REF_IMG = cfg['reference']
    METHOD = cfg['method']
    PLOT = cfg['plot']

    os.chdir(WORKING_DIR)

    # Create an output directory in case it doesn't exist
    output_dir = os.path.join(OUTPUT_DIR, METHOD)
    misc.ensure_dir_exists(output_dir)

    with rio.open(REF_IMG) as src:
        reference = src.read()

    raster_files = glob(os.path.join(INPUT_DIR, '*.tif'))
    for img in tqdm(raster_files, desc='Homogeneise colour histogram', total=len(raster_files)):
        with rio.open(img) as src:
            image = src.read()

        if METHOD == 'scikit':
            matched = exposure.match_histograms(image, reference, channel_axis=0)
        elif METHOD == 'rasterio':
            matched = histogram_match(image, reference)

        output_meta = src.meta.copy()
        output_meta.update(
        {"driver": "GTiff",
            "height": matched.shape[1],
            "width": matched.shape[2],
            "crs": src.crs
        })

        img_name = os.path.basename(img)
        output_path = os.path.join(OUTPUT_DIR, METHOD, img_name)
        with rio.open(output_path, 'w', **output_meta) as dst:
            dst.write(matched)

        if PLOT:
            image_p = image.T
            reference_p = reference.T
            matched_p = matched.T
            # plot_image(image_p, reference_p, matched_p, output_dir, img_name)
            plot_histogram(image_p, reference_p, matched_p, output_dir, img_name)