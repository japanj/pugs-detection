"""
utils.py

This module contains common utility functions used throughout the project,
such as printing basic information about GeoDataFrame and setting random seeds
for reproducibility.

Author: Pitchaporn Likitpanjamanon
Date: 01-05-2025
"""

import numpy as np
import random
import torch
import os
from lightning.pytorch import seed_everything
from IPython.display import display


def print_basic_info(gdf):
    """
    Print basic information about GeoDataFrame.

    Parameters
    ----------
    gdf : GeoDataFrame
        The GeoDataFrame
    """
    # Print the basic information
    print("Number of rows:", len(gdf))
    print("Number of columns:", len(gdf.columns))
    display(gdf.head(3))


def set_all_seeds(seed=42):
    """
    Set all random seeds for reproducibility

    Parameters:
    ----------
    seed : int
        The seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    seed_everything(42, workers=True)

def get_checkpoint_path(version):
    checkpoint_dir = f"../models/trained_models/version_{version}/checkpoints"
    files = os.listdir(checkpoint_dir)
    
    if not files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    
    checkpoint_file = files[0]
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    
    print(f"Found checkpoint: {checkpoint_file}")
    return checkpoint_path