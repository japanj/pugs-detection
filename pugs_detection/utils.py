"""
utils.py

This module contains utility functions for processing geospatial data,
including OSM data and Sentinel-2 image processing.

Author: Pitchaporn Likitpanjamanon
Date: [YYYY-MM-DD]
"""

import numpy as np
import torch
from lightning.pytorch import seed_everything
from IPython.display import display

def print_basic_info(gdf):
    """
    Print basic information about the OSM data.

    Parameters
    ----------
    gdf : GeoDataFrame
        The OSM data as a GeoDataFrame.
    """
    # Print the basic information
    print("Number of rows:", len(gdf))
    print("Number of columns:", len(gdf.columns))
    display(gdf.head())

def set_all_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    seed_everything(42, workers=True)