# Public Urban Green Spaces Detection

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Public Urban Green Spaces Detection Workflow

## Overview of the workflow

## Project Organization

```
├── LICENSE             <- Open-source license if one is chosen
├── Makefile            <- Makefile with convenience commands like `make data` or `make train`
├── README.md           <- The top-level README for developers using this project.
├── data
│   ├── processed       <- The intermediate data from raw data processing.
│   └── raw             <- The original, immutable data dump.
│
├── docs                <- A default mkdocs project; see www.mkdocs.org for details
│
├── models              <- Trained and serialized (saved) models, model evaluations, and
│   │                      model hyperparameters log
│   ├── checkpoints     <- Best model files and training logs for each experiment
│   └── test_result     <- Model performance metrics from the best model
│
├── notebooks           <- Jupyter notebooks are named using the following convention:
│                          `<step_number>_<short_description>.ipynb`, e.g. `1_data_acquisition_osm`.
│
├── pyproject.toml      <- Project configuration file with package metadata for 
│                          public_urban_green_spaces_detection and configuration for tools like black
│
├── references          <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports             <- Generated analysis (e.g. HTML, PDF, LaTeX, etc.)
│   └── figures         <- Generated graphics and figures to be used in reporting
│       └── loss graph  <- Training and validation loss graph for each experiment
│
├── environment.yml     <- The requirements file for reproducing the analysis environment, e.g.
│                          generated with `pip freeze > requirements.txt`
│
├── setup.cfg           <- Configuration file for flake8
│
└── pugs_detection      <- Source code for use in this project.
    │
    ├── modeling                
    │   ├── __init__.py
    │   ├── model.py                 <- Customize the model architecture
    │   ├── predict.py               <- Code to run model inference with trained models          
    │   ├── segmentation_task.py     <- Customize SegmentationTask from torchgeo library
    │   └── evaluation.py            <- Code to run create the confusion matrix
    │
    ├── __init__.py          <- Makes public_urban_green_spaces_detection a Python module
    ├── utils.py             <- Store functions used within data processing step
    ├── dataset.py           <- Scripts to download or generate data
    ├── ground_truth.py      <- Code to create features for modeling
    ├── osm.py               <-
    ├── raster.py            <-
    └── plots.py             <- Code to create all visualizations
```

## Datasets

## Hardware and Software (e.g. OS) Spec

I use Windows Subsystem for Linux (WSL) to work on and run the workflow.
```
Host Operating System (OS): Windows 11 (64-bit OS, x64-based processor)
Workflow Environment: WSL2 (Ubuntu 22.04.3 LTS)
CPU: 13th Gen Intel(R) Core(TM) i7-13700H 2.40 GHz
GPU: NVIDIA RTX A500
RAM: 32 GB
Disk storage: 1 TB
```

## Set up the environment

## Steps to run the workflow

## Notebook viewer
| Notebook name         | Link to view notebook | Description           |
| --------------------- | --------------------- | --------------------- |
| 2_data_processing_osm.ipynb | https://nbviewer.org/github/japanj/pugs-detection/blob/dev/notebooks/2_data_processing_osm.ipynb | Processing OpenStreetMap data which is used in model training step for PUGS detection |
| new_ground_truth_exploration.ipynb  | https://nbviewer.org/github/japanj/pugs-detection/blob/dev/notebooks/new_ground_truth_exploration.ipynb | Exploration of ground truth datasets |
| ground_truth_creation.ipynb  | https://nbviewer.org/github/japanj/pugs-detection/blob/dev/notebooks/ground_truth_creation.ipynb | Processing ground truth datasets and create one single ground truth |

## License
