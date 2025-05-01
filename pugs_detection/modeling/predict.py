"""
predict.py

This module contains functions for predicting the whole area using a trained model.

Author: Pitchaporn Likitpanjamanon
Date: 01-05-2025
"""

import numpy as np
import torch
from tqdm import tqdm

def predict_whole_area(model, inference_dataset, inference_loader):
    """
    Predict the whole area using a trained model.

    Parameters:
    -----------
    model : CustomSegmentationTask
        Trained model to be used for inference
    inference_dataset : PredictedImageDataset
        Dataset containing the images to be predicted
    inference_loader : DataLoader
        DataLoader for the inference dataset
    
    Returns:
    --------
    prediction_map : numpy array
        Array containing the predicted values for the whole area
    """
    # Create output array
    prediction_map = np.zeros(
        (inference_dataset.height, inference_dataset.width), dtype=np.float32
    )
    counts = np.zeros((inference_dataset.height, inference_dataset.width), dtype=np.float32)

    # Process all patches
    model.eval()
    with torch.no_grad():
        for batch in tqdm(inference_loader):
            # Get images and window info
            images = batch["image"]
            windows = batch["window_info"]

            # Run inference
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            # Process each patch in the batch
            for i, window in enumerate(windows):
                # Get window coordinates
                x_off = window.col_off
                y_off = window.row_off
                width = window.width
                height = window.height

                # Get prediction for this patch
                pred = preds[i].squeeze().cpu().numpy()
                # Ensure prediction is 2D for plotting
                if pred.ndim > 2:
                    if pred.shape[0] == 1:
                        # Single channel but in shape (1, H, W)
                        pred = pred.squeeze(0)
                    else:
                        pred = pred[1]
                # Add to output array
                prediction_map[y_off : y_off + height, x_off : x_off + width] += pred
                counts[y_off : y_off + height, x_off : x_off + width] += 1

    # Average overlapping predictions
    with np.errstate(divide="ignore", invalid="ignore"):
        prediction_map = np.divide(prediction_map, counts)
        # prediction_map = np.nan_to_num(prediction_map)
        # Add thresholding to force binary output
        prediction_map = (prediction_map > 0.5).astype(np.uint8)

    return prediction_map