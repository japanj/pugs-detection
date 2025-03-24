"""
features.py

This module contains functions for feature engineering, 
including image normalization and patch generation for deep learning model.

Author: Pitchaporn Likitpanjamanon
Date: [YYYY-MM-DD]
"""

# put normalization later

import torchgeo
import torch
import torch.nn as nn
import rioxarray
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import copy
from torchgeo.datasets import RasterDataset, VectorDataset
from torchgeo.samplers import GridGeoSampler
from torch.utils.data import Dataset
from pyproj import CRS
from pugs_detection.utils import set_all_seeds

def create_image_tiles(output_folder_path, image_path, train_index_list, val_index_list, test_index_list):
    # Check directory existence
    try:
        os.makedirs(output_folder_path, exist_ok=True)
    except FileExistsError:
        # Directory already exists
        pass

    # Open the raster data
    rds = rioxarray.open_rasterio(image_path)

    # Get the bounds of the image
    xmin, ymin, xmax, ymax = rds.rio.bounds()

    # Calculate the width and height of each tile
    print("width:", xmax - xmin, "height:", ymax - ymin)
    base_tile_width = (xmax - xmin) // 5
    base_tile_height = (ymax - ymin) // 5
    print("tile width:", base_tile_width, "tile height:", base_tile_height)

    # Create 25 tiles (5x5 grid) with special handling for the last column/row
    tiles = []
    for i in range(5):
        for j in range(5):
            # For columns 0-3, use regular spacing
            tile_xmin = xmin + i * base_tile_width
            
            # For the last column, extend to the edge
            if i == 4:
                tile_xmax = xmax
            else:
                tile_xmax = xmin + (i + 1) * base_tile_width
            
            # For the last row, extend to the edge
            tile_ymin = ymin + j * base_tile_height
            if j == 4:
                tile_ymax = ymax
            else:
                tile_ymax = ymin + (j + 1) * base_tile_height
            
            tiles.append([tile_xmin, tile_xmax, tile_ymin, tile_ymax])

    # Save image tiles to different output folders
    for idx, tile in enumerate(tiles):
        tile_xmin, tile_xmax, tile_ymin, tile_ymax = tile

        # Clip the raster to this tile and save it
        if (idx+1) in train_index_list:
            subfolder_path = os.path.join(output_folder_path, 'train')
            os.makedirs(subfolder_path, exist_ok=True)
        elif (idx+1) in val_index_list:
            subfolder_path = os.path.join(output_folder_path, 'val')
            os.makedirs(subfolder_path, exist_ok=True)
        else:
            subfolder_path = os.path.join(output_folder_path, 'test')
            os.makedirs(subfolder_path, exist_ok=True)

        tile_rds = rds.rio.clip_box(minx=tile_xmin, miny=tile_ymin, maxx=tile_xmax, maxy=tile_ymax)
        tile_file_path = os.path.join(subfolder_path, f'tile_{idx + 1}.geotiff')
        tile_rds.rio.to_raster(tile_file_path, driver='GTiff')
        print(f"Tile {idx + 1} saved to {tile_file_path}")

    return tiles

def _process_mask(mask, valid_area, band_count):
    mask = mask.numpy()
    valid_area = valid_area.numpy()
    # change the valid area array position to be Red band (important for vegetation detection so I won't replace it)
    if band_count >= 13:
        valid_area_all = valid_area[0] | valid_area[1] | valid_area[2] | valid_area[3] | valid_area[4] | valid_area[5] | valid_area[6] | valid_area[7] | valid_area[8] | valid_area[9] | valid_area[10] | valid_area[11] | valid_area[12]
    else:
        valid_area_all = valid_area[0] | valid_area[1] | valid_area[2] | valid_area[3] | valid_area[4] | valid_area[5] | valid_area[6] | valid_area[7] | valid_area[8]
    # mask[~valid_area[2]] = 0
    mask[~valid_area_all] = 0
    result = torch.from_numpy(mask)
    return result

# Create a custom filtering function
def _filter_patches(sample):
    """Filter patches that contain only background or only PUGS"""
    mask = sample['mask'].numpy()
    # Calculate percentage of green space in the mask
    green_percentage = np.mean(mask)
    
    # Keep patches that have between 5% and 95% green space
    return 0.05 < green_percentage < 0.95

# Custom dataset class that applies filtering
class FilteredGeoDataset(Dataset):
    # stride is set as 64 for both x and y directions
    def __init__(self, dataset, patch_size=256, stride=64, transform=None, specific_bands=list(range(13))):
        self.dataset = dataset
        self.sampler = GridGeoSampler(dataset, size=(patch_size, patch_size), stride=stride)
        self.transform = transform
        self.bounds = dataset.bounds
        self.specific_bands = specific_bands
        self.band_count = len(specific_bands)
        
        # Compute the valid patches
        self.valid_bboxes = []
        count = 0
        for bbox in self.sampler:
            sample = self.dataset[bbox]
            if _filter_patches(sample):
                self.valid_bboxes.append(bbox)
            count += 1
            print(count)
        print(f"Found {len(self.valid_bboxes)} valid patches out of {len(self.sampler)} total patches")
    
    def __len__(self):
        return len(self.valid_bboxes)
    
    def __getitem__(self, idx):
        sample = self.dataset[self.valid_bboxes[idx]]

        # Select specific bands    
        sample['image'] = sample['image'][self.specific_bands]
        min_value = sample['image'].min()
        valid_area = (sample['image']!=min_value) # create a mask of valid area

        # sample['image'], valid_area = contrast_stretch_patch(sample['image'])
        sample['mask'] = _process_mask(sample['mask'], valid_area, self.band_count)
        
        
        del sample["crs"]
        del sample["bounds"]

        return sample
    
class AugmentedDataset(Dataset):
    """Dataset wrapper that applies multiple augmentations to increase sample count"""
    def __init__(self, dataset, transform_list=None):
        self.dataset = dataset
        self.transform_list = transform_list
    
    def __len__(self):
        return len(self.dataset) * (len(self.transform_list) + 1)  # Original + augmented versions
    
    def __getitem__(self, idx):
        # Calculate original dataset index and augmentation version
        dataset_idx = idx // (len(self.transform_list) + 1)
        aug_version = idx % (len(self.transform_list) + 1)
        
        # Get original sample
        sample = self.dataset[dataset_idx]
        
        # If it's the first version (aug_version=0), return original
        if aug_version == 0 or self.transform_list is None:
            return sample
        
        # Apply transformation
        transform = self.transform_list[aug_version - 1]

        # Make a deep copy to avoid modifying the original
        sample_copy = copy.deepcopy(sample)
        
        # Apply the transformation
        sample_copy = transform(sample_copy)

        # Ensure proper shape (remove batch dimension added by some transforms)
        sample_copy['image'] = sample_copy['image'].squeeze(0)
        sample_copy['mask'] = sample_copy['mask'].squeeze(0)
            
        return sample_copy
    
# Create datasets for each split
def create_dataset_split(image_path, label_path, epsg_code, band_list, dataset_type, transform_list):
    set_all_seeds(42)
    image_ds = RasterDataset(paths=image_path, crs=CRS.from_epsg(epsg_code), res=10)
    label_ds = VectorDataset(paths=label_path, crs=CRS.from_epsg(epsg_code), res=10)
    combined_ds = image_ds & label_ds
    filter_ds = FilteredGeoDataset(dataset=combined_ds, stride=128, specific_bands=band_list)
    if dataset_type == 'train':
        return AugmentedDataset(dataset=filter_ds, transform_list=transform_list)
    else:
        return filter_ds