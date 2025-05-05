"""
plots.py

This module contains functions for plotting results
and visualizing data.

Author: Pitchaporn Likitpanjamanon
Date: 01-05-2025
"""

import rioxarray
import rasterio
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import folium
import pandas as pd
import seaborn as sns
import os
from matplotlib.ticker import MaxNLocator
from rasterio.windows import Window
from tqdm import tqdm
from pugs_detection.utils import set_all_seeds


def plot_image_tiles(image_path, image_tiles):
    """
    Plot the image with bounding boxes for each tile.

    Parameters:
    -----------
    image_path : str
        Path to the satellite image
    image_tiles : list
        List that contains tile information (e.g. xmin, xmax, ymin, ymax)
    """
    data = rioxarray.open_rasterio(image_path)
    plt.figure(figsize=(12, 12))
    data.sel(band=[4, 3, 2]).plot.imshow(robust=True)
    plt.axis("off")

    # Add bounding boxes for each tile
    ax = plt.gca()
    for idx, tile in enumerate(image_tiles):
        if tile != 0:
            tile_xmin, tile_xmax, tile_ymin, tile_ymax, valid_pct = tile
            width = tile_xmax - tile_xmin
            height = tile_ymax - tile_ymin

            # Create a rectangle patch
            rect = patches.Rectangle(
                (tile_xmin, tile_ymin),
                width,
                height,
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )

            # Add the rectangle to the plot
            ax.add_patch(rect)

            # Add text label for the tile number
            plt.text(
                tile_xmin + width / 2,
                tile_ymin + height / 2,
                f"Tile {idx + 1}",
                ha="center",
                va="center",
                color="white",
                fontsize=10,
                bbox=dict(facecolor="black", alpha=0.7, boxstyle="round,pad=0.2"),
            )

    plt.title("Satellite Image with tiles")
    plt.show()


def enhance_satellite_rgb(img):
    """
    Enhance RGB visualization of satellite imagery.

    Parameters:
    -----------
    img : numpy array
        Multi-band satellite image with shape [bands, height, width]

    Returns:
    --------
    rgb : numpy array
        Enhanced RGB image with values from 0-1, shape [height, width, 3]
    """
    # Extract RGB bands (Sentinel-2 convention: R=band 4, G=band 3, B=band 2)
    if img.shape[0] >= 13:
        r = img[3].copy()
        g = img[2].copy()
        b = img[1].copy()
    else:
        r = img[2].copy()
        g = img[1].copy()
        b = img[0].copy()

    rgb_list = [r, g, b]
    valid_mask = (r > 0) & (g > 0) & (b > 0)
    # Process each band individually
    for i in range(3):
        band = rgb_list[i]
        # Only consider valid pixels for percentile calculation
        if np.sum(valid_mask) > 0:  # Check if we have valid pixels
            valid_pixels = band[valid_mask]
            lower, upper = np.percentile(valid_pixels, (2, 98))
        else:
            # Fallback if no valid pixels found
            lower, upper = np.percentile(band, (2, 98))

        # Apply contrast stretch
        rgb_list[i] = np.clip((band - lower) / (upper - lower), 0, 1)

    # Stack bands into RGB
    rgb = np.stack(rgb_list, axis=2)

    return rgb


def visualize_from_torchgeo_dataloader(
    dataloader, num_samples=3, mode="original", additional_band_pos=None
):
    """
    Visualize samples of input channels or input images from Dataloader

    Parameters:
    -----------
    dataloader : DataLoader
        Dataloader containing the dataset
    num_samples : int
        Number of samples to visualize
    mode : str
        Mode of visualization ('original' or 'additional')
        Note: 'additional' is when the additional data from OSM is added
    additional_band_pos : list
        List of additional information from OSM band position
    """
    set_all_seeds(42)
    # Get a batch from the dataloader
    dataiter = iter(dataloader)
    batch = next(dataiter)

    for i in range(min(num_samples, batch["image"].shape[0])):
        # Get the image and mask for this sample
        image = batch["image"][i]
        mask = batch["mask"][i]

        # Convert to numpy for visualization
        image_np = image.numpy()
        mask_np = mask.numpy()

        # Print information about the sample
        print(f"Sample {i}:")
        print(f"  Image shape: {image_np.shape}")
        print(f"  Mask shape: {mask_np.shape}")
        print(f"  Green space percentage: {np.mean(mask_np):.2f}")

        rgb = enhance_satellite_rgb(image_np)

        if mode != "original":
            num_plots = 2 + len(additional_band_pos)
        else:
            num_plots = 2

        # Create a figure with two subplots
        fig, axes = plt.subplots(1, num_plots, figsize=(12, 6))

        additional_info_pos = 0
        for i in range(num_plots):
            if i == 0:
                # Display the RGB image in the first subplot
                axes[i].imshow(rgb)
                axes[i].set_title("RGB Image")
                axes[i].axis("off")
            elif i == (num_plots - 1):
                axes[i].imshow(mask_np, cmap="gray")
                axes[i].set_title("GT")
                axes[i].axis("off")
            else:
                # Display the mask in the second subplot
                axes[i].imshow(
                    image_np[additional_band_pos[additional_info_pos]], cmap="gray"
                )
                axes[i].set_title("Additional info")
                axes[i].axis("off")
                additional_info_pos += 1

        plt.show()


def visualize_map(
    gdf_list,
    column_list=None,
    name_list=None,
    tooltip_list=None,
    style_list=None,
    tile_list=None,
    marker_style=None,
    categorical=False,
):
    """
    Visualize multiple GeoDataFrames on a Folium map

    Parameters:
    -----------
    gdf_list : list
        List of GeoDataFrames to visualize
    column_list : list
        List of columns to use for coloring the layers
    name_list : list
        List of names for the layers
    tooltip_list : list
        List of tooltips for the layers
    style_list : list
        List of style dictionaries for the layers
    tile_list : list
        List of tile layers to add to the map
    marker_style : list
        List of marker styles for the layers
    categorical : bool
        Whether the column values are categorical
    """
    # Create the base map with the first layer
    m = gdf_list[0].explore(
        column=column_list[0] if column_list is not None else None,
        name=name_list[0] if name_list is not None else None,
        style_kwds=style_list[0] if style_list is not None else {},
        tooltip=tooltip_list[0] if tooltip_list is not None else True,
        marker_kwds=marker_style[0] if marker_style is not None else {},
        categorical=categorical,
    )

    for i in range(1, len(gdf_list)):
        # Add each GeoDataFrame to the map
        gdf_list[i].explore(
            m=m,
            column=column_list[i] if column_list is not None else None,
            name=name_list[i] if name_list is not None else None,
            style_kwds=style_list[i] if style_list is not None else {},
            tooltip=tooltip_list[i] if tooltip_list is not None else True,
            marker_kwds=marker_style[i] if marker_style is not None else {},
            categorical=categorical,
        )

    if tile_list is not None:
        # Add tile layers if provided
        for tile in tile_list:
            if tile == "OSM":
                folium.TileLayer(
                    tiles="OpenStreetMap",
                    name="OpenStreetMap",
                    overlay=False,
                    control=True,
                ).add_to(m)
            else:
                folium.TileLayer(
                    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                    attr="Esri",
                    name="Esri Satellite",
                    overlay=False,
                    control=True,
                ).add_to(m)

    # Add layer controls
    folium.LayerControl().add_to(m)

    return m


def visualize_predictions(
    model,
    test_loader,
    num_batches=5,
    samples_per_batch=4,
    mode="original",
    replace_band_pos=None,
    output_dir=None,
):
    """
    Visualize predictions from a model on test data

    Parameters:
    -----------
    model : CustomSegmentationTask
        The trained model
    test_loader : DataLoader
        DataLoader for the test dataset
    num_batches : int
        Number of batches to visualize
    samples_per_batch : int
        Number of samples to visualize per batch
    mode : str
        Mode of visualization ('original' or 'additional')
        Note: 'additional' is when the additional data from OSM is added
    additional_band_pos : list
        List of additional information from OSM band position
    output_dir : str
        Directory to save the output images
    """
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
            # logits = model.model.predict(images)
            logits = model(images)
            # Ensure predictions have the right shape
            preds = torch.sigmoid(logits)
            preds = (preds > 0.5).float()

            # Print shapes for debugging
            print(f"Prediction shape: {preds.shape}")
            print(f"Mask shape: {masks.shape}")

        # Create visualization
        num_samples = min(samples_per_batch, images.shape[0])
        if mode != "original":
            num_plots = 3 + len(replace_band_pos)
        else:
            num_plots = 3

        fig, axes = plt.subplots(num_samples, num_plots, figsize=(15, 5 * num_samples))

        if num_samples == 1:
            axes = axes.reshape(1, 3)

        for i in range(num_samples):
            # Get sample
            img = images[i].cpu().numpy()
            mask = masks[i].cpu().numpy()
            pred = preds[i].cpu().numpy()

            print(
                f"Sample {i} - Image shape: {img.shape}, Mask shape: {mask.shape}, Pred shape: {pred.shape}"
            )

            rgb = enhance_satellite_rgb(img)
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

            additional_info_pos = 0
            # Plot
            for j in range(num_plots):
                if j == 0:
                    print(np.percentile(rgb, 2), np.percentile(rgb, 98))
                    axes[i, j].imshow(rgb)
                    axes[i, j].set_title("Image")
                    axes[i, j].axis("off")
                elif j == 1:
                    axes[i, j].imshow(mask, cmap="gray")
                    axes[i, j].set_title("Ground Truth")
                    axes[i, j].axis("off")
                elif j == (num_plots - 1):
                    try:
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
                    axes[i, j].imshow(
                        img[replace_band_pos[additional_info_pos]], cmap="gray"
                    )
                    axes[i, j].set_title("Additional Info")
                    axes[i, j].axis("off")
                    additional_info_pos += 1

        plt.tight_layout()

        # Save the figure if output_dir is provided
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            fig_path = os.path.join(output_dir, f"batch_{batch_idx}.png")
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")

        # Show the figure
        plt.show()

        # Close the figure to free memory
        plt.close(fig)


def visualize_area(
    original_image_path,
    prediction_path,
    ground_truth_path=None,
    x_coord=0,
    y_coord=0,
    window_size=256,
):
    """
    Visualize a specific area with RGB, ground truth, and prediction.

    Parameters:
    -----------
    original_image_path : str
        Path to the satellite image
    prediction_path : str
        Path to the binary mask from model prediction
    ground_truth_path : str
        Path to the binary mask from ground truth
    x_coord, y_coord : int
        Coordinates of the top left corner of the area to visualize
    window_size : int
        Size of the window to extract (square)
    """
    # Open original image
    with rasterio.open(original_image_path) as src:
        window = Window(x_coord, y_coord, window_size, window_size)
        original = src.read(window=window)
        # print(f"Original shape: {original.shape}")

    # Open prediction
    with rasterio.open(prediction_path) as src:
        prediction = src.read(1, window=window)
        pred_nodata = src.nodata
        if pred_nodata is not None:
            prediction = np.where(
                prediction == pred_nodata, 0, prediction
            )  # Set nodata to 0
        # print(f"Prediction shape: {prediction.shape}")

    # Load ground truth if provided
    ground_truth = None
    if ground_truth_path and os.path.exists(ground_truth_path):
        try:
            with rasterio.open(ground_truth_path) as src:
                ground_truth = src.read(1, window=window)
                gt_nodata = src.nodata
                if gt_nodata is not None:
                    ground_truth = np.where(
                        ground_truth == gt_nodata, 0, ground_truth
                    )  # Set nodata to 0
                # print(f"Ground truth shape: {ground_truth.shape}")
        except Exception as e:
            print(f"Error loading ground truth: {e}")

    # Create RGB visualization
    rgb_data = enhance_satellite_rgb(original)

    # Determine number of plots
    # if ground_truth is not None:
    #     fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    #     axes = axes.flatten()
    # else:
    #     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    # Plot RGB
    axes[0].imshow(rgb_data)
    axes[0].set_title("RGB Image", fontsize=14)
    axes[0].axis("off")

    # # Plot ground truth if available
    if ground_truth is not None:
        axes[1].imshow(ground_truth, cmap="gray")
        axes[1].set_title("Ground Truth", fontsize=14)
        axes[1].axis("off")
        idx_pred = 2  # Index for prediction plot
    else:
        idx_pred = 1  # Index for prediction plot when no ground truth

    # Plot prediction
    axes[idx_pred].imshow(prediction, cmap="gray")
    axes[idx_pred].set_title("Model Prediction", fontsize=14)
    axes[idx_pred].axis("off")

    plt.tight_layout()
    return fig


def visualize_whole_image(
    original_image_path,
    prediction_path,
    ground_truth_path=None,
    window_size=256,
    stride=256,
    output_dir=None,
):
    """
    Loop through the entire image and visualize multiple regions.

    Parameters:
    -----------
    original_image_path : str
        Path to the satellite image
    prediction_path : str
        Path to the binary mask from model prediction
    ground_truth_path : str
        Path to the binary mask from ground truth
    window_size : int
        Size of each window to extract
    stride : int
        Step size when moving through the image
    output_dir : str, optional
        Directory to save visualizations (if None, just displays them)
    """
    # Get image dimensions
    with rasterio.open(original_image_path) as src:
        img_width = src.width
        img_height = src.height

    # Create output directory if needed
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Calculate grid positions
    positions = []
    for y in range(0, img_height - window_size + 1, stride):
        for x in range(0, img_width - window_size + 1, stride):
            positions.append((x, y))

    print(f"Found {len(positions)} windows to visualize")

    # Create visualizations
    for i, (x, y) in enumerate(tqdm(positions)):
        try:
            fig = visualize_area(
                original_image_path=original_image_path,
                prediction_path=prediction_path,
                ground_truth_path=ground_truth_path,
                x_coord=x,
                y_coord=y,
                window_size=window_size,
            )

            if output_dir:
                # Save to file
                fig_path = os.path.join(output_dir, f"region_{x}_{y}.png")
                fig.savefig(fig_path, bbox_inches="tight", dpi=300)
                plt.close(fig)
            else:
                # Display and pause
                plt.show()
                response = input(
                    f"Window {i + 1}/{len(positions)} at ({x},{y}). Press Enter to continue, 'q' to quit: "
                )
                if response.lower() == "q":
                    break
                plt.close(fig)

        except Exception as e:
            print(f"Error visualizing window at ({x},{y}): {e}")

    return f"Completed visualization of {len(positions)} windows"


def plot_loss_graph(metrics_path, loss_graph_path):
    """
    Plot training and validation loss from a CSV file
    and save the figure

    Parameters:
    -----------
    metrics_path : str
        Path to the CSV file containing training and validation metrics (e.g. train_loss, val_loss)
    loss_graph_path : str
        Path to save the loss graph
    """
    # Read the metrics CSV file
    df = pd.read_csv(metrics_path)

    # Create a figure with appropriate size
    fig = plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    # Plot mean training loss per epoch
    train_df = df.groupby("epoch")["train_loss"].mean().reset_index()
    plt.plot(
        train_df["epoch"],
        train_df["train_loss"],
        "b-",
        marker="o",
        markersize=4,
        linewidth=2,
        label="Training Loss (Mean)",
    )

    # Plot validation loss - filter rows where val_loss exists
    val_df = df[df["val_loss"].notna()].copy()
    plt.plot(
        val_df["epoch"],
        val_df["val_loss"],
        "r-",
        marker="o",
        markersize=6,
        linewidth=2,
        label="Validation Loss",
    )

    # Add labels and title
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title("Training and Validation Loss Over Epochs", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Adjust axis for better visualization
    plt.xlim(left=0)
    plt.ylim(bottom=0)

    # Set x-axis to integer ticks only
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Add vertical line at best validation performance
    best_epoch = val_df.loc[val_df["val_loss"].idxmin()]["epoch"]
    plt.axvline(
        x=best_epoch,
        color="g",
        linestyle="--",
        alpha=0.7,
        label=f"Best model (epoch {int(best_epoch)})",
    )

    plt.tight_layout()
    plt.savefig(loss_graph_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # Print best epoch stats
    best_row = val_df[val_df["epoch"] == best_epoch].iloc[0]
    print(f"Best model at epoch {int(best_epoch)}:")
    print(f"  Val Loss: {best_row['val_loss']:.4f}")
    print(f"  Val Jaccard: {best_row['val_BinaryJaccardIndex']:.4f}")
