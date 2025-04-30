"""
evaluation.py

This module contains functions for generating confusion matrix.

Author: Pitchaporn Likitpanjamanon
Date: [YYYY-MM-DD]
"""

from sklearn.metrics import confusion_matrix
import numpy as np
import torch


def generate_confusion_matrix(model, test_loader):
    """
    Generate confusion matrix

    Parameters:
    -----------
    model : numpy array
        Multi-band satellite image with shape [bands, height, width]

    Returns:
    --------
    rgb : numpy array
        Enhanced RGB image with values from 0-1, shape [height, width, 3]
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
