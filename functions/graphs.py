import sys
from loguru import logger

import pygeohash as pgh
import networkx as nx

from functions.misc import format_logger

logger = format_logger(logger)



def assign_groups(row, group_index, column):
    """Assign a group number to GT and detection of a geodataframe

    Args:
        row (row): geodataframe row

    Returns:
        row (row): row with a new 'group_id' column
    """

    try:
        row['group_id'] = group_index[getattr(row, column)]
    except: 
        row['group_id'] = None
    
    return row



def make_groups(gdf, column_left='geohash_left', column_right='geohash_right'):
    """Identify groups based on pairing nodes with NetworkX. The Graph is a collection of nodes.
    Nodes are hashable objects (geohash (str)).

    Returns:
        groups (list): list of connected geohash groups
    """

    g = nx.Graph()
    for row in gdf[gdf[column_left].notnull()].itertuples():
        g.add_edge(getattr(row, column_left), getattr(row, column_right))

    groups = list(nx.connected_components(g))

    return groups


