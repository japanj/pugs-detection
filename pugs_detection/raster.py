"""
raster.py

This module contains functions for processing raster data,
including processing the pixel values and
creating image tiles, binary mask, and signed-distance transform raster.

Author: Pitchaporn Likitpanjamanon
Date: [YYYY-MM-DD]
"""

import rioxarray
import os
import rasterio
import numpy as np
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt

def create_image_tiles(output_folder_path, image_path, train_index_list, val_index_list, test_index_list, valid_data_threshold=0.3):
    """
    Create train/val/test tiles from a satellite image, filtering out tiles with insufficient valid data.
    
    Parameters:
    -----------
    output_folder_path : str
        Path to save output tiles
    image_path : str
        Path to the satellite image (possibly clipped)
    train_index_list : list
        List of tile indices for training set
    val_index_list : list
        List of tile indices for validation set
    test_index_list : list
        List of tile indices for test set
    valid_data_threshold : float
        Minimum percentage of valid data required in a tile (0.0-1.0)
    """
    # Check directory existence
    try:
        os.makedirs(output_folder_path, exist_ok=True)
    except FileExistsError:
        # Directory already exists
        pass

    # Create subdirectories
    train_dir = os.path.join(output_folder_path, 'train')
    val_dir = os.path.join(output_folder_path, 'val')
    test_dir = os.path.join(output_folder_path, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Open the raster data with masked=True for proper no-data handling
    rds = rioxarray.open_rasterio(image_path, masked=True)

    # Get the bounds of the image
    xmin, ymin, xmax, ymax = rds.rio.bounds()

    # Calculate the width and height of each tile
    print("Image bounds width:", xmax - xmin, "height:", ymax - ymin)
    base_tile_width = (xmax - xmin) // 5
    base_tile_height = (ymax - ymin) // 5
    print("Tile dimensions width:", base_tile_width, "height:", base_tile_height)

    tiles = []
    valid_tile_count = 0
    skipped_tile_count = 0
    
    for i in range(5):
        for j in range(5):
            # Calculate tile boundaries
            tile_xmin = xmin + i * base_tile_width
            tile_ymin = ymin + j * base_tile_height
            
            # For the last column/row, extend to the edge
            tile_xmax = xmax if i == 4 else xmin + (i + 1) * base_tile_width
            tile_ymax = ymax if j == 4 else ymin + (j + 1) * base_tile_height
            
            # Generate tile index (1-25)
            idx = i * 5 + j
            
            try:
                # Clip the raster to this tile
                tile_rds = rds.rio.clip_box(
                    minx=tile_xmin, miny=tile_ymin, 
                    maxx=tile_xmax, maxy=tile_ymax
                )
                
                # Check for valid data percentage (non-masked values)
                # Use the first band to create a mask
                valid_mask = ~tile_rds[0].isnull().values
                valid_percentage = np.sum(valid_mask) / valid_mask.size
                
                # Skip tile if it doesn't have enough valid data
                if valid_percentage < valid_data_threshold:
                    print(f"Skipping tile {idx+1}: only {valid_percentage:.1%} valid data")
                    skipped_tile_count += 1
                    tiles.append(0)
                    continue
                
                # Determine which folder to save in
                if (idx+1) in train_index_list:
                    subfolder_path = train_dir
                    split_type = "train"
                elif (idx+1) in val_index_list:
                    subfolder_path = val_dir
                    split_type = "val"
                else:
                    subfolder_path = test_dir
                    split_type = "test"
                
                # Save the tile
                tile_file_path = os.path.join(subfolder_path, f'tile_{idx + 1}.geotiff')
                
                # Save with float preservation and proper no-data values
                tile_rds.rio.to_raster(
                    tile_file_path, 
                    driver='GTiff',
                    dtype='float32'
                )
                
                # Add tile to the list of valid tiles
                tiles.append([tile_xmin, tile_xmax, tile_ymin, tile_ymax, valid_percentage])
                valid_tile_count += 1
                
                print(f"Tile {idx + 1} saved to {split_type} folder ({valid_percentage:.1%} valid data)")
                
            except Exception as e:
                print(f"Error processing tile {idx+1}: {e}")
                skipped_tile_count += 1
    
    print(f"Processing complete: {valid_tile_count} valid tiles created, {skipped_tile_count} tiles skipped")
    return tiles

def create_binary_mask(gdf, file_path, satellite_image_path):
    with rasterio.open(satellite_image_path) as src:
        transform = src.transform
        width = src.width
        height = src.height
        crs = src.crs

        # Create an empty raster with the same properties as the satellite image
        out_shape = (height, width)
        raster = np.zeros(out_shape, dtype='uint8')

        # Rasterize the polygons
        rasterized = rasterize(
            [(geom, 1) for geom in gdf.geometry],
            out_shape=out_shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype='uint8'
        )
        
        # Save the raster to a file
        with rasterio.open(
            file_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=raster.dtype,
            crs=gdf.crs,
            transform=transform,
        ) as dst:
            dst.write(rasterized, 1)

    print("Rasterization complete.")

def create_sdt_raster(gdf, file_path, satellite_image_path):
    with rasterio.open(satellite_image_path) as src:
        transform = src.transform
        width = src.width
        height = src.height

        # Create an empty raster with the same properties as the satellite image
        out_shape = (height, width)
        raster = np.zeros(out_shape, dtype='uint8')
        
        # Rasterize the polygons
        raster = rasterize(
            [(geom, 1) for geom in gdf.geometry],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            all_touched=True,
            dtype='uint8'
        )

        # Compute the distance transform for the inside and outside of the polygons
        distance_inside = distance_transform_edt(raster) # Calculate the nearest distance from each pixel inside polygon to the nearest pixel outside polygon
        distance_outside = distance_transform_edt(1 - raster) # Calculate the nearest distance from each pixel outside polygon to the nearest pixel inside polygon

        # Create the signed distance transform
        signed_distance_transform = distance_inside - distance_outside

        # Save the signed distance transform to a file
        with rasterio.open(
            file_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=signed_distance_transform.dtype,
            crs=gdf.crs,
            transform=transform,
        ) as dst:
            dst.write(signed_distance_transform, 1)

        print("Signed distance transform complete.")

def contrast_stretch(array, lower_percentile=1, upper_percentile=99):
    normalized_img = array.copy()
    
    for i in range(array.sizes['band']):
        # Get the band index and data
        band_idx = array.band.values[i]  # Use actual band coordinate value
        band = array.sel(band=band_idx).values
        
        # Calculate percentiles
        lower = np.percentile(band, lower_percentile)
        upper = np.percentile(band, upper_percentile)

        print(f"Band {i+1}: Lower {lower}, Upper {upper}")

        if upper > lower:
            normalized_img.loc[dict(band=band_idx)] = np.clip((band - lower) / (upper - lower), 0, 1)
        else:
            normalized_img.loc[dict(band=band_idx)] = np.zeros_like(band)
    
    return normalized_img

def contrast_stretch_dn(array):
    result = array.copy()
    # Apply a percentile-based contrast stretch (ignore the extreme values so image isn't too dark or too bright)
    array_new = np.where(array.values < 0, 0, array.values)
    image_norm = array_new / 10000
    image_norm = np.clip(image_norm, 0, 1)
    result.values = image_norm
    return result

# Min-max scaling to [-1, 1] range
def normalize_sdt_minmax(sdt_array):
    sdt_min = float(sdt_array.min())
    sdt_max = float(sdt_array.max())
    # Apply min-max formula scaled to [-1, 1]
    normalized = 2 * (sdt_array - sdt_min) / (sdt_max - sdt_min) - 1
    return normalized

def min_max_normalize(data):
    """Scale data to 0-1 range using min-max normalization"""
    min_val = data.min().values
    max_val = data.max().values
    return (data - min_val) / (max_val - min_val)