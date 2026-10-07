"""
evaluation.py

This module contains functions for generating confusion matrix.

Author: Pitchaporn Likitpanjamanon/Rosa Aguilar
Date: 05-10-2026
"""

import geopandas as gpd
import pandas as pd
from unittest import result
from rasterstats import zonal_stats

from sklearn.metrics import confusion_matrix
from pugs_detection.ground_truth import merge_overlapping_polygons
import numpy as np
import torch


def generate_confusion_matrix(model, test_loader):
    """
    Generate confusion matrix

    Parameters:
    -----------
    model : CustomSegmentationTask
        Model to be evaluated

    Returns:
    --------
    cm : numpy array
        Confusion matrix of the predictions
    """
    # Set model to evaluation mode
    model.eval()

    # Collect all predictions and ground truths
    all_preds = []
    all_masks = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"]
            masks = batch["mask"]

            # Generate predictions
            logits = model(images)
            preds = torch.sigmoid(logits)
            preds_binary = (preds > 0.5).float()

            # Add to collection (flatten everything)
            all_preds.extend(preds_binary.cpu().numpy().flatten())
            all_masks.extend(masks.cpu().numpy().flatten())

    # Calculate confusion matrix
    cm = confusion_matrix(all_masks, all_preds)

    return cm

def calculate_perinstance_recall(gt_vector, predicted_raster):

    
    gt_gdf = gpd.read_file(gt_vector)
    merged_gdf = merge_overlapping_polygons(gt_gdf, threshold=0.5)

    # Calculate area and group by size
    merged_gdf["area_m2"] = merged_gdf.geometry.area
    # Assign bins and labels for area groups
    bins = [0, 4000, 30000, 100000, 800000, merged_gdf["area_m2"].max()]
    labels = ["<= 0.4", "(0.4, 3]", "(3, 10]", "(10, 80]", "> 80"]
    merged_gdf["area_group"] = pd.cut(merged_gdf["area_m2"], bins=bins, labels=labels, include_lowest=True)
    
    stats_in_gt = zonal_stats(
    merged_gdf,  # vector
    predicted_raster,  # raster
    stats=["mean", "count"],
    nodata=-9999,  # treat -9999 as nodata
   )

    stats_in_gt_df = pd.DataFrame(stats_in_gt)

    stats_in_gt_df = pd.concat(
        [merged_gdf[["geometry","area_group", "area_m2"]], stats_in_gt_df], axis=1
    )
    COV_THR = 0.5
    stats_in_gt_df['detected'] = stats_in_gt_df["mean"].fillna(0) >= COV_THR
    per_class_detection = (stats_in_gt_df.groupby("area_group", observed=True)["detected"]
                        .agg(n="size", n_detected="sum", recall="mean").reset_index()
                        )
   
    return per_class_detection