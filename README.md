# Public Urban Green Spaces (PUGS) Detection Workflow

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This project provides a reproducible workflow for PUGS detection, using open-source technology, by integrating Sentinel-2 image and OpenStreetMap (OSM) data with a deep learning method.

The study area in this project is **Dresden, Germany**.

## Overview of the workflow

This workflow encompass from data acquisition to model evaluation step. The final output of this workflow is the **binary mask of PUGS** in the study or target area.

![Flowchart of the workflow](reports/figures/diagram/update_main_workflow.jpg)

The model architecture used for detecting PUGS is **U-Net with ResNet-50 backbone**. The model used pre-trained weight from TorchGeo library. More details about model training can be found in [Model folder document](/models/README.md).

## Project Organization
```
├── LICENSE             <- Open-source license of the project
├── README.md           <- The top-level README for users.
├── data                <- All input datasets used in the workflow, including both raw data 
│   │                      and processed data generated during data preparation steps
│   ├── processed       <- The intermediate data from raw data processing.
│   └── raw             <- The original, immutable data dump.
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
├── environment.yml     <- The requirements file for reproducing the analysis environment
│
└── pugs_detection      <- Main Python package containing modules and utilities for PUGS detection workflow
    │
    ├── modeling                
    │   ├── __init__.py
    │   ├── model.py                 <- Customized model architecture
    │   ├── predict.py               <- Code to run model inference with trained models          
    │   ├── segmentation_task.py     <- Customized SegmentationTask (customize from torchgeo library)
    │   └── evaluation.py            <- Function to create the confusion matrix
    │
    ├── __init__.py          
    ├── utils.py             <- Common functions used across different notebooks
    ├── dataset.py           <- Dataset classes and dataset creation functions
    ├── ground_truth.py      <- Functions to create features for modeling
    ├── osm.py               <- Functions related to OSM data including loading and processing OSM data
    ├── raster.py            <- Functions to create and process raster data
    └── plots.py             <- Functions to create all visualizations
```

## Datasets
| Main input data      | Source         | Description        | Data License       |
| -------------------- | -------------- | ------------------ | ------------------ |
| Sentinel-2 image | Copernicus Data Space Ecosystem | TBD | The data is regulated under EU law (Commission Delegated Regulation (EU) No 1159/2013) which based on a principle of full, open and free access. <br> More details about data policy: https://sentinel.esa.int/documents/247904/690755/Sentinel_Data_Legal_Notice |
| OSM data | OSM | TBD | [Open Data Commons Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/) |
| Ground truth data | European Union's Copernicus Land Monitoring Service | TBD | The data is regulated under EU law (Commission Delegated Regulation (EU) No 1159/2013) which based on a principle of full, open and free access. <br> More details about data policy: https://land.copernicus.eu/en/data-policy |
| Ground truth data | Dresden Open Data Portal | TBD | [dl-de/by-2-0](https://www.govdata.de/dl-de/by-2-0) |
| Ground truth data | Author | TBD | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) |

More details about input data can be found in [Data folder documentation](/data/README.md)

## Hardware and Software Specifications
Windows Subsystem for Linux (WSL) is used to develop and run the workflow.
```
Host Operating System (OS): Windows 11 (64-bit OS, x64-based processor)
Workflow Environment: WSL2 (Ubuntu 22.04.3 LTS)
CPU: 13th Gen Intel(R) Core(TM) i7-13700H 2.40 GHz
GPU: NVIDIA RTX A500
RAM: 32 GB
Disk storage: 1 TB
```

## Prerequisites
1. User need to have Conda installed. If user have not installed Conda yet, please visit the [installation guide](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html) from Conda.
2. User need to clone this repository or download ZIP file of this repository. To clone the repository, user can use the below command.
    ```
    git clone https://github.com/japanj/pugs-detection.git
    ```

## Set up the environment
1. Create the environment with required dependencies for this project

    ```
    conda env create -f environment.yml
    ```
2. Activate the environment
    ```
    conda activate pugs-detection
    ```
(Optional) User can check all the installed libraries or dependencies by using `conda list` command. 

## Steps to run the workflow
All the notebooks are available in `notebooks` folder. The **number at the beginning of the notebooks' name indicate the order of execution**. Each notebook has different requirements or dependecies which are described in the notebooks.

To get the reproducible result in data processing till model evaluation result step, user can skip the data acquisition notebooks (`1_data_acquisition_ground_truth.ipynb`, `1_data_acquisition_osm.ipynb`, and `1_data_acquisition_satellite_image.ipynb`) since there might be a chance the **different data retrieving time** can lead to slightly different dataset, e.g. OSM data.

## Results
(Attach the result table, example of prediction output, some figure?, link to analysis notebook?)

## License
It can be separated into three sections:
1. This repository is released under [MIT License](/LICENSE).
2. The data used in this project is under various licenses. Please visit [Data license section](/data/README.md#data-license) to see more detail about license of data used in this project.
3. The model weights, prediction output, and all figures are licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/legalcode.en).

## Contact
If there is any further questions or issues related to work, please feel free to open an issues in GitHub or contact m.p.likitpanjamanon@student.utwente.nl