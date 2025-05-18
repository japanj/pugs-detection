# pugs_detection folder

This folder contains the main Python package for Public Urban Green Spaces (PUGS) detection workflow. It implements the core functionality used in the project, including data processing, model architecture modification, model prediction, model evaluation, and visualization components.

## Folder structure
```
├── README.md      <- README file for pugs_detection folder
│
├── modeling                
│   ├── __init__.py
│   ├── model.py                <- Customized model architecture
│   ├── predict.py              <- Code to run model inference with trained models          
│   ├── segmentation_task.py    <- Customized SegmentationTask (customize from torchgeo library)
│   └── evaluation.py           <- Function to create the confusion matrix
│
├── __init__.py          
├── utils.py           <- Common functions used across different notebooks
├── dataset.py         <- Dataset classes and dataset creation functions
├── ground_truth.py    <- Functions to create features for modeling
├── osm.py             <- Functions related to OSM data including loading and processing OSM data
├── raster.py          <- Functions to create and process raster data
└── plots.py           <- Functions to create all visualizations