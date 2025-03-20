"""
plots.py

This module contains functions for plotting results 
and visualizing data.

Author: Pitchaporn Likitpanjamanon
Date: [YYYY-MM-DD]
"""

import rioxarray
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pugs_detection.utils import set_all_seeds

def plot_image_tiles(image_path, image_tiles):
    data = rioxarray.open_rasterio(image_path)
    # Plot the original satellite image with bounding boxes for the 16 tiles
    plt.figure(figsize=(12, 12))
    data.sel(band=[4, 3, 2]).plot.imshow(robust=True)
    plt.axis('off')

    # Add bounding boxes for each tile
    ax = plt.gca()
    for idx, tile in enumerate(image_tiles):
        tile_xmin, tile_xmax, tile_ymin, tile_ymax = tile
        width = tile_xmax - tile_xmin
        height = tile_ymax - tile_ymin
        
        # Create a rectangle patch
        rect = patches.Rectangle((tile_xmin, tile_ymin), width, height, 
                                linewidth=1, edgecolor='r', facecolor='none')
        
        # Add the rectangle to the plot
        ax.add_patch(rect)
        
        # Add text label for the tile number
        plt.text(tile_xmin + width/2, tile_ymin + height/2, f'Tile {idx+1}', 
                ha='center', va='center', color='white', fontsize=10,
                bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.2'))

    plt.title(f"Satellite Image with {len(image_tiles)} Tiles")
    plt.show()

def visualize_from_torchgeo_dataloader(dataloader, num_samples=3, mode='original', replace_band_pos=None):
    """Visualize samples from a TorchGeo DataLoader with stack_samples"""
    set_all_seeds(42)
    # Get a batch from the dataloader
    dataiter = iter(dataloader)
    batch = next(dataiter)

    for i in range(min(num_samples, batch['image'].shape[0])):
        # Get the image and mask for this sample
        image = batch['image'][i]
        mask = batch['mask'][i]
        
        # Convert to numpy for visualization
        image_np = image.numpy()
        mask_np = mask.numpy()

        # Print information about the sample
        print(f"Sample {i}:")
        print(f"  Image shape: {image_np.shape}")
        print(f"  Mask shape: {mask_np.shape}")
        print(f"  Green space percentage: {np.mean(mask_np):.2f}")
        
        # For sentinel-2 data, use bands 4,3,2 (R,G,B) or 8,4,3 (NIR,R,G)
        # Assuming bands are [C, H, W]
        if image_np.shape[0] > 3:  # Multi-spectral image
            rgb = image_np[[3, 2, 1], :, :].transpose(1, 2, 0)  # Select 4,3,2 bands (RGB bands)
            # rgb = image_np[[2, 1, 0], :, :].transpose(1, 2, 0)  # Select 4,3,2 bands (RGB bands)
        else:
            rgb = image_np.transpose(1, 2, 0)
        

        if mode != 'original':
            image_additional_mask = image_np[replace_band_pos]
            num_plots = 3
        else:
            num_plots = 2
        
        # Create a figure with two subplots
        fig, axes = plt.subplots(1, num_plots, figsize=(12, 6))
        
        for i in range(num_plots):
            if i==0:
                # Display the RGB image in the first subplot
                axes[i].imshow(rgb)
                axes[i].set_title('RGB Image')
                axes[i].axis('off')
            elif i==(num_plots-1):
                axes[i].imshow(mask_np, cmap='gray')
                axes[i].set_title('GT')
                axes[i].axis('off')
            else:
                # Display the mask in the second subplot
                axes[i].imshow(image_additional_mask, cmap='gray')
                axes[i].set_title('Additional info')
                axes[i].axis('off')
        
        plt.show()