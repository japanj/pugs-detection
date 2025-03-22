"""
model.py

This module contains functions for model architecture modification.

Author: Pitchaporn Likitpanjamanon
Date: [YYYY-MM-DD]
"""

import torchgeo
import torch
import torch.nn as nn

def modify_first_layer(task, in_channels=14):    
    # Get the model from the task
    model = task.model
    
    # Find the first conv layer (can delete first if -> hasattr(model, 'backbone'))
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'conv1'):
        # Some other architecture
        first_conv = model.encoder.conv1
    # else:
    #     # If structure is different, try to find the first conv
    #     for module in model.modules():
    #         if isinstance(module, nn.Conv2d) and module.in_channels == 13:
    #             first_conv = module
    #             break
    
    # Store the original weights
    original_weights = first_conv.weight.data
    
    # Create a new conv with more input channels
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=first_conv.bias is not None
    )

    if in_channels >= 13:
        # Copy the original weights for the first 13 channels
        with torch.no_grad():
            new_conv.weight[:, :13] = original_weights
            # Initialize the new channel(s) with small random values
            if in_channels > 13:
                nn.init.kaiming_normal_(new_conv.weight[:, 13:], mode='fan_out', nonlinearity='relu')
                # nn.init.kaiming_normal_(new_conv.weight[:, 13:])
    else:
        # Copy the original weights for the first 13 channels
        with torch.no_grad():
            new_conv.weight[:, :10] = original_weights
            # Initialize the new channel(s) with small random values
            if in_channels > 9:
                nn.init.kaiming_normal_(new_conv.weight[:, 9:], mode='fan_out', nonlinearity='relu')
    
    # Replace the original conv with the new one
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'conv1'):
        model.encoder.conv1 = new_conv
    else:
        # If we found it through modules() earlier
        raise ValueError("Could not replace the first convolutional layer")
    
    print(task)
    
    return task