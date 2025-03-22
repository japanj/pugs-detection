"""
plots.py

This module contains functions for plotting results 
and visualizing data.

Author: Pitchaporn Likitpanjamanon
Date: [YYYY-MM-DD]
"""

import rioxarray
import numpy as np
import torch
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
        
        # For sentinel-2 data, use bands 4,3,2 (R,G,B)
        # Assuming bands are [C, H, W]
        if image_np.shape[0] >= 13:  # Multi-spectral image
            rgb = image_np[[3, 2, 1], :, :].transpose(1, 2, 0)  # Select 4,3,2 bands (RGB bands)
        else:
            rgb = image_np[[2, 1, 0], :, :].transpose(1, 2, 0)  # Select 4,3,2 bands (RGB bands)
        

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

def visualize_predictions(model, test_loader, num_batches=5, samples_per_batch=4, mode='original'):
    # Set model to evaluation mode
    model.eval()
    
    # Iterate through batches
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= num_batches:
            break
        
        # Get data
        images = batch["image"]
        masks = batch["mask"]
        
        # Generate predictions
        with torch.no_grad():
            logits = model(images)
            # Ensure predictions have the right shape
            preds = torch.sigmoid(logits)
            preds = (preds > 0.5).float()
            
            # Print shapes for debugging
            print(f"Prediction shape: {preds.shape}")
            print(f"Mask shape: {masks.shape}")
        
        # Create visualization
        num_samples = min(samples_per_batch, images.shape[0])
        if mode != 'original':
            num_plots = 4
        else:
            num_plots = 3
        
        fig, axes = plt.subplots(num_samples, num_plots, figsize=(15, 5*num_samples))

        if num_samples == 1:
            axes = axes.reshape(1, 3)
        
        for i in range(num_samples):
            # Get sample
            img = images[i].cpu().numpy()
            mask = masks[i].cpu().numpy()
            pred = preds[i].cpu().numpy()
            
            print(f"Sample {i} - Image shape: {img.shape}, Mask shape: {mask.shape}, Pred shape: {pred.shape}")

            # Create RGB visualization for the image
            if img.shape[0] >= 13:
                # Sentinel-2 RGB composite
                rgb = np.stack([img[3], img[2], img[1]], axis=0)
            else:
                # Sentinel-2 RGB composite
                rgb = np.stack([img[2], img[1], img[0]], axis=0)
            rgb = np.transpose(rgb, (1, 2, 0))
            
            # Ensure mask is 2D for plotting
            if mask.ndim > 2:
                if mask.shape[0] == 1:
                    # Single channel but in shape (1, H, W)
                    mask = mask.squeeze(0)
                else:
                    # Multiple channels, take the first one
                    mask = mask[0]
            
            # Ensure prediction is 2D for plotting
            if pred.ndim > 2:
                if pred.shape[0] == 1:
                    # Single channel but in shape (1, H, W)
                    pred = pred.squeeze(0)
                else:
                    pred = pred[1]
                    print(np.max(pred))
            
            # Plot
            for j in range(num_plots):
                if j==0:
                    axes[i, j].imshow(rgb)
                    axes[i, j].set_title("Image")
                    axes[i, j].axis("off")
                if j==1:
                    axes[i, j].imshow(mask, cmap="gray")
                    axes[i, j].set_title("Ground Truth")
                    axes[i, j].axis("off")
                elif j==(num_plots-1):
                    try:
                        # axes[i, 3].imshow(pred, cmap="gray")
                        # axes[i, 3].set_title("Prediction")
                        # axes[i, 3].axis("off")
                        axes[i, j].imshow(pred, cmap="gray")
                        axes[i, j].set_title("Prediction")
                        axes[i, j].axis("off")
                    except Exception as e:
                        print(f"Error plotting prediction: {e}")
                        print(f"Prediction shape: {pred.shape}, dtype: {pred.dtype}")
                        # Try alternate approach
                        if pred.ndim > 2:
                            pred_2d = pred.mean(axis=0)  # Average across channels
                            axes[i, j].imshow(pred_2d, cmap="gray")
                        else:
                            # Create a blank image
                            blank = np.zeros_like(mask)
                            axes[i, j].imshow(blank, cmap="gray")
                        axes[i, j].set_title("Prediction (Error)")
                        axes[i, j].axis("off")
                else:
                    axes[i, j].imshow(img[9], cmap="gray")
                    axes[i, j].set_title("Additional Info")
                    axes[i, j].axis("off")

        plt.tight_layout()
        plt.show()