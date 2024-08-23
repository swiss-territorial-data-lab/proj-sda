import os
import sys
from loguru import logger


def format_logger(logger):
    """Format the logger from loguru

    Args:
        logger: logger object from loguru

    Returns:
        logger: formatted logger object
    """

    logger.remove()
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}",
            level="INFO", filter=lambda record: record["level"].no < 25)
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - <green>{level}</green> - {message}",
            level="SUCCESS", filter=lambda record: record["level"].no < 30)
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - <yellow>{level}</yellow> - {message}",
            level="WARNING", filter=lambda record: record["level"].no < 40)
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - <red>{level}</red> - <level>{message}</level>",
            level="ERROR")

    return logger


def ensure_dir_exists(dirpath):
    """Test if a directory exists. If not, make it.  

    Args:
        dirpath (str): directory path to test

    Returns:
        dirpath (str): directory path that have been tested
    """

    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
        logger.info(f"The directory {dirpath} was created.")
    
    return dirpath


def overlap(polygon1_shape, polygon2_shape):
    """Determine the overlap area of one polygon with another one

    Args:
        polygon1_shape (geometry): first polygon
        polygon2_shape (geometry): second polygon

    Returns:
        float: ratio of overlapped area
    """

    # Calculate intersection area
    polygon_intersection = polygon1_shape.intersection(polygon2_shape).area
    polygon_area = polygon1_shape.area
    
    return polygon_intersection / polygon_area