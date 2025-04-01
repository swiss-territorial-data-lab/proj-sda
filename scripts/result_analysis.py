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
import functions.misc as misc
from functions.constants import DONE_MSG

from loguru import logger
logger = misc.format_logger(logger)


def count_values(df, attribut_to_count, count_name, step=1):
    """
    Count the number of times each value appears in a given dataframe attribute.

    Args
    ----------
    df : pandas.DataFrame
        The dataframe to count the values from.
    attribut_to_count : str
        The attribute to count the values of.
    count_name : str
        The name of the new attribute that will contain the count.
    step : int
        The step to use for the values to count. Default is 1.

    Returns
    -------
    pandas.DataFrame
        A dataframe with the count of each value in the given attribute.
    """
    count_per_attribute_df = df.value_counts(attribut_to_count).reset_index().rename(
        columns={'count': count_name}
    )
    all_steps = pd.Series(range(count_per_attribute_df[attribut_to_count].min(), count_per_attribute_df[attribut_to_count].max()+step, step), name='steps')
    all_count_per_attribute_df = pd.merge(
        count_per_attribute_df, all_steps, how='outer', left_on=attribut_to_count, right_on='steps'
    ).sort_values('steps').reset_index(drop=True)
    all_count_per_attribute_df.loc[all_count_per_attribute_df[count_name].isna(), count_name] = 0

    return all_count_per_attribute_df

def format_barplot(df, ax, fig, labels, title, plot_path, xlabel='Year', rotation=45):
    """
    Format a barplot with labels and title.

    Args
    ----------
    df : pandas.DataFrame
        The dataframe containing the data to plot.
    ax : matplotlib.axes
        The axes where the plot is drawn.
    fig : matplotlib.figure
        The figure containing the axes.
    labels : list
        The labels for the x-axis.
    title : str
        The title of the plot.
    plot_path : str
        The path where to save the plot.
    xlabel : str
        The label for the x-axis. Default is 'Year'.
    rotation : int
        The rotation of the labels on the x-axis. Default is 45.

    Returns
    -------
    None
    """
    
    plt.gca().set_yticks(plt.gca().get_yticks().tolist())
    ticks_to_use = df.index[::5]

    ax.set_xticks(ticks_to_use, labels, rotation=rotation, fontsize=10, ha='center')
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    plt.xlabel(xlabel, fontweight='bold')

    plt.title(title, fontweight='bold')
    plt.legend(loc='upper left', frameon=False)    

    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)
    

def format_scatterplot(fig, canton, title, ylabel, xlabel='Year', output_dir='graphs'):
    """
    Format and save a scatter plot.
    
    Args
    ----------
    fig : matplotlib.figure
        The figure containing the axes.
    canton : str
        The name of the canton.
    title : str
        The title of the plot.
    ylabel : str
        The label for the y-axis.
    xlabel : str, optional
        The label for the x-axis. Default is 'Year'.
    output_dir : str, optional
        The directory where to save the plot. Default is 'graphs'.

    Returns
    -------
    str
        The path where the plot was saved.
    """
    
    plt.xlabel(xlabel, fontweight='bold')
    plt.ylabel(ylabel, fontweight='bold')

    plt.title(title, fontweight='bold')

    plot_path = os.path.join(output_dir, f'{ylabel.lower().replace(" ", "_")}_per_{xlabel.lower().replace(" ", "_")}_{canton}.jpg')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)

    return plot_path

def plot_barchart(df, cat, min_year, max_year, class_dict, data, output_dir):
    """
    Plot a bar chart with number of TP, FN and FP for a given category 'cat' in the given years range.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the data to plot.
    cat : str
        The category to plot.
    min_year : int
        The minimum year to plot.
    max_year : int
        The maximum year to plot.
    class_dict : dict, optional
        A dictionary with class names as keys and their corresponding display names as values. Default is None.
    data : str, optional
        The data to plot. Can be 'label', 'det' or 'both'. Default is 'label'.
    output_dir : str, optional
        The directory where to save the plot. Default is 'graphs'.
    
    Returns
    -------
    str
        The path where the plot was saved.
    """
    
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

    labels = df[year][::5]
    title = class_dict[cat] if class_dict else cat
    plot_path = os.path.join(output_dir, f'{data}_{title}.png'.replace(' ', '_'))
    format_barplot(df, ax, fig, labels, title, 'all_cantons', plot_path)

    return plot_path


def plot_boxplot(df, param, output_dir):
    """
    Plot a boxplot of the given parameter for each tag in the given dataframe.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the data to plot.
    param : str
        The parameter to plot.
    output_dir : str
        The directory where to save the plot.
    
    Returns
    -------
    str
        The path where the plot was saved.
    """

    ax = df.boxplot(column=[param], by=['tag'], grid=False)
    if param == 'area':
        ax.set_yscale('symlog')
        ax.yaxis.set_major_formatter(lambda x, p: f'{int(x):,}')
    
    ax.set_xlabel('Detection tags', fontweight='bold')
    ax.set_ylabel(param.capitalize(), fontweight='bold')
    ax.get_figure().suptitle('')
    ax.set_title('')

    plot_path = os.path.join(output_dir, f'{param}_boxplot.png')
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
    WORKING_DIR = cfg['working_directory']
    OUTPUT_DIR = cfg['output_directory']
    DETECTIONS = cfg['detections']

    os.chdir(WORKING_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    written_files = [] 

    # Convert input detections to a geodataframe 
    logger.info('Read data...')
    detections_gdf = gpd.read_file(DETECTIONS)
    detections_gdf = detections_gdf.to_crs(2056)

    total = len(detections_gdf)
    logger.info(f"{total} input shapes")

    # Training results
    if 'tag' in detections_gdf.columns:
        MIN_YEAR = cfg['min_year'] if 'min_year' in cfg.keys() else 1945
        MAX_YEAR = cfg['max_year'] if 'max_year' in cfg.keys() else 2025
        CLASS_DICT = cfg['class_dict'] if 'class_dict' in cfg.keys() else None
        detections_gdf['area'] = detections_gdf.area 
        for parameter in ['area', 'score']:
            feature = plot_boxplot(detections_gdf, param=parameter, output_dir=OUTPUT_DIR)
            written_files.append(feature)
            logger.success(f"{DONE_MSG} A file was written: {feature}") 

        for cat in filter(None, detections_gdf.CATEGORY.unique()): 
            feature = plot_barchart(detections_gdf, cat, MIN_YEAR, MAX_YEAR, CLASS_DICT, data='label', output_dir=OUTPUT_DIR)
            written_files.append(feature)
            feature = plot_barchart(detections_gdf, cat, MIN_YEAR, MAX_YEAR, CLASS_DICT, data='det', output_dir=OUTPUT_DIR)
            written_files.append(feature) 
            feature = plot_barchart(detections_gdf, cat, MIN_YEAR, MAX_YEAR, CLASS_DICT, data='both', output_dir=OUTPUT_DIR)
            written_files.append(feature)

    else:   # Inference results
        if 'vaud' in DETECTIONS or 'vaud' in WORKING_DIR:
            CANTON = 'vaud'
        elif 'ticino' in DETECTIONS or 'ticino' in WORKING_DIR:
            CANTON = 'ticino'
        else:
            logger.critical(f'No canton indicator in the detection path.')
            sys.exit(1)

        all_years = pd.Series(range(detections_gdf.year_det.min(), detections_gdf.year_det.max()+1), name='year')

        logger.info('Plot total surface and score per year...')
        dets_per_year = detections_gdf[['year_det', 'score', 'merged_score', 'valid_area', 'geometry']].dissolve(['year_det'], aggfunc=np.median, as_index=False)
        dets_per_year['covered area [km2]'] = dets_per_year.area/1000000
        full_dets_per_year = pd.merge(all_years, dets_per_year, left_on='year', right_on='year_det', how='outer').sort_values('year')
        full_dets_per_year.loc[full_dets_per_year['covered area [km2]'].isna(), 'covered area [km2]'] = 0
        
        # Barplot of the total surface
        plt.rcParams["figure.figsize"] = (len(all_years)*0.15, 5)
        fig, ax = plt.subplots(1, 1)
        ax = full_dets_per_year.plot(x='year', y='covered area [km2]', kind='bar', rot=0, log=False, width=0.75, grid=True)

        labels = full_dets_per_year.year[::5].astype(int)
        plot_path = os.path.join(OUTPUT_DIR, f'surface_per_year_{CANTON}.jpg')
        format_barplot(full_dets_per_year, ax, fig, labels, 'Area covered with detection for each year', plot_path)
        written_files.append(plot_path)

        # Scatterplot of the median merged score per year
        plt.rcParams["figure.figsize"] = (12, 5)
        fig, ax = plt.subplots(1, 1)
        ax = full_dets_per_year.plot(x='year', y='merged_score', kind='scatter', rot=0, grid=True)
        written_files.append(format_scatterplot(
            fig, CANTON, 'Median merged score for each year', 'Median merged score', output_dir=OUTPUT_DIR
        ))

        # Scatterplot of the median area per year
        plt.rcParams["figure.figsize"] = (12, 5)
        fig, ax = plt.subplots(1, 1)
        ax = full_dets_per_year.plot(x='year', y='valid_area', kind='scatter', rot=0, grid=True)
        written_files.append(format_scatterplot(
            fig, CANTON, 'Median area per det for each year', 'Median area per detection', output_dir=OUTPUT_DIR
        ))

        # Boxplot of the merged score per yeaer
        plt.rcParams["figure.figsize"] = (12, 5)
        fig, ax = plt.subplots(1, 1)
        ax = full_dets_per_year.boxplot(column=['merged_score'], by=['year'], grid=True)

        plt.gca().set_yticks(plt.gca().get_yticks().tolist())
        labels = full_dets_per_year.year[::5].astype(int)
        ticks_to_use = full_dets_per_year.index[::5]

        ax.set_xticks(ticks_to_use, labels, rotation=0, fontsize=10, ha='center')
        ax.xaxis.set_minor_locator(MultipleLocator(1))
    
        ax.set_xlabel('Detection tags', fontweight='bold')
        ax.set_ylabel('merged_score'.replace('_', ' ').capitalize(), fontweight='bold')
        ax.get_figure().suptitle('')
        ax.set_title('')

        plot_path = os.path.join(OUTPUT_DIR, f'merged_score_boxplot.png')
        plt.savefig(plot_path, bbox_inches='tight')

        # Scatterplot of the median score per year
        plt.rcParams["figure.figsize"] = (12, 5)
        fig, ax = plt.subplots(1, 1)
        ax = full_dets_per_year.plot(x='year', y='score', kind='scatter', rot=0, grid=True)
        written_files.append(format_scatterplot(
            fig, CANTON, 'Median score for each year', 'Median score', output_dir=OUTPUT_DIR
        ))

        del dets_per_year, full_dets_per_year

        logger.info('Plot the number of dets per year...')
        all_count_per_year_df = count_values(detections_gdf, 'year_det', 'number of detections')
        plt.rcParams["figure.figsize"] = (len(all_years)*0.15, 5)
        fig, ax = plt.subplots(1, 1)
        ax = all_count_per_year_df.plot(x='steps', y='number of detections', kind='bar', rot=0, log=False, width=0.75, grid=True)

        labels = all_count_per_year_df.steps[::5].astype(int)
        plot_path = os.path.join(OUTPUT_DIR, f'detections_per_year_{CANTON}.jpg')
        format_barplot(all_count_per_year_df, ax, fig, labels, 'Number of detections per year', plot_path)

        logger.info('Plot the amount of dets present in multiple years...')
        all_count_per_year_df = count_values(detections_gdf, 'count_years', 'number of detections')
        plt.rcParams["figure.figsize"] = (len(all_years)*0.1, 5)
        fig, ax = plt.subplots(1, 1)
        ax = all_count_per_year_df.plot(x='steps', y='number of detections', kind='bar', rot=0, log=True, width=0.5, grid=True)

        labels = all_count_per_year_df.steps[::5].astype(int)
        plot_path = os.path.join(OUTPUT_DIR, f'several_appearance_through_years_{CANTON}.jpg')
        format_barplot(
            all_count_per_year_df, ax, fig, labels, 'Number of detections detected one or several times', plot_path,
            xlabel='Number of appearance', rotation=0
        )
        written_files.append(plot_path)
        del all_count_per_year_df

        logger.info('Plot the number of dets per elevation bin...')
        detections_gdf['rounded_elevation'] = [25*round(elev/25) for elev in detections_gdf.elevation]
        all_count_per_elevation_df = count_values(detections_gdf, 'rounded_elevation', 'number of detections', step=25)

        plt.rcParams["figure.figsize"] = (len(all_years)*0.15, 5)
        fig, ax = plt.subplots(1, 1)
        ax = all_count_per_elevation_df.plot(x='steps', y='number of detections', kind='bar', rot=0, log=False, width=0.5, grid=True)

        labels = all_count_per_elevation_df.steps[::5].astype(int)
        plot_path = os.path.join(OUTPUT_DIR, f'detections_per_elevation_{CANTON}.jpg')
        format_barplot(
            all_count_per_elevation_df, ax, fig, labels, 'Number of detections per elevation rounded to 25 m', plot_path,
            xlabel='Elevation', rotation=45
        )
        written_files.append(plot_path)
        del all_count_per_elevation_df
        
    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(os.path.join(WORKING_DIR, written_file))

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()