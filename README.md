# Public Urban Green Spaces (PUGS) Detection Workflow

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This project provides a reproducible workflow for PUGS detection, using open-source technology, by integrating Sentinel-2 image and OpenStreetMap (OSM) data with a deep learning method.

The study area in this project is **Dresden, Germany**.

## Overview of the workflow

This workflow covers all steps from data acquisition to model evaluation and result analysis. The final output is the **binary mask of PUGS** for the study area.

![Flowchart of the workflow](reports/figures/diagram/update_main_workflow.jpg)

The model architecture used for detecting PUGS is **U-Net with ResNet-50 backbone**. The model uses pre-trained weight from [TorchGeo library](https://github.com/microsoft/torchgeo). See [Model documentation](/models/README.md) for more details about model setup.

## Project Organization
```
├── LICENSE             <- Open-source license of the project
├── README.md           <- The top-level README for users
├── config.yaml         <- All parameters used in the workflow
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
├── results             <- Results from model prediction
│   ├── clipped_prediction     <- Model prediction results clipped to Dresden administrative boundary 
│   ├── fn_maps                <- False negative (FN) area maps from model prediction
│   ├── fp_maps                <- False positive (FP) area maps from model prediction 
│   └── whole_area_prediction  <- Model prediction results
│
├── reports             <- Generated analysis (e.g. HTML, PDF, LaTeX, etc.)
│   ├── figures         <- Generated graphics and figures to be used in reporting
│   └── notebook html   <- Notebooks in HTML format
│   
├── pyproject.toml      <- Project metadata
├── uv.lock             <- Lockfile that contains information about project's dependencies
│
└── pugs_detection      <- Main Python package containing modules and utilities for PUGS detection workflow
```

## Datasets
| Main input data      | Source         | Description        | Data License       |
| -------------------- | -------------- | ------------------ | ------------------ |
| Sentinel-2 image | Copernicus Data Space Ecosystem | High-resolution and multi-spectral satellite image. It includes 13 bands and spatial resolution is 10m, 20m, and 60m depending on the wavelength | The data is regulated under EU law (Commission Delegated Regulation (EU) No 1159/2013) which based on a principle of full, open and free access. <br> More details about data policy: [Sentinel-2 Data Policy](https://sentinel.esa.int/documents/247904/690755/Sentinel_Data_Legal_Notice) |
| OSM data | OSM | Areas/polygons related to green spaces, Point of Interest (POI) such as bench, and footpath network | [Open Data Commons Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/) |
| Ground truth data | European Union's Copernicus Land Monitoring Service (CLMS) | Provide land cover and land use data in Functional Urban Areas (FUA) (This dataset is downloaded by specifying only Dresden area) | The data is regulated under EU law (Commission Delegated Regulation (EU) No 1159/2013) which based on a principle of full, open and free access. <br> More details about data policy: [CLMS Data Policy](https://land.copernicus.eu/en/data-policy) |
| Ground truth data | Dresden Open Data Portal | Datasets related to PUGS | [dl-de/by-2-0](https://www.govdata.de/dl-de/by-2-0) |
| Ground truth data | Author | Manually digitized boundary of Park Zwirnmühle | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) |

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
**uv** (Python package and project manager) is used in this project. The version of **uv** is 0.7.3.


## Estimated project size
The project size is approximately 5.7 GB.
- data: approximately 1.64 GB
- models: approximately 3.27 GB
- code and others: approximately 0.79 GB

## Getting Started

1. Clone this repository or download ZIP file of this repository. To clone the repository, use the following command:
    ```
    git clone https://github.com/japanj/pugs-detection.git
    ```
2. Download **data and model checkpoints** of all experiments from https://doi.org/10.5281/zenodo.15553942

   > Make sure that data and models are placed in the same folder structure in [Project Organization](#project-organization)

3. Make sure that you have installed `Python` and `pip`.

4. Install **uv** package by running
   ```
   pip install uv
   ```
   *Note:* `pipx install uv` is also recommened as it will automatically installed  uv package in isolated environment. You can visit [uv document](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) to see more details about uv installation.

   To install exact version of **uv**, you can use the following command `pip install uv==0.7.3` or `pipx install uv==0.7.3`.

5. Navigate to the root directory of the cloned repository. You should be in the directory containing `pyproject.toml`.

6. Set up the virtual environment by running
   ```
   uv sync
   ```
   The virtual environment (`.venv` folder) will be automatically created in project folder.

7. You can verify the installed packages by running
   ```
   uv pip list
   ```

### Steps to run the workflow
All notebooks are available in [notebooks folder](/notebooks/). The **number at the beginning of each notebook's name indicates the execution order**. Each notebook has different requirements or dependencies which are described in the notebooks.

**Note:**
- On **Windows and macOS**, some parameters in [config.yaml](/config.yaml) are required to change due to OS differences. Please follow the instruction in [Reproducibility guide](reproducibility.md).

## Results
The result of different model training experiments are shown in the table below. To see the detail of Model experiment setup, please visit [Model Experiment Setup document](/models/README.md#model-experiment-setup).
| Version folder | input  | Loss function | Jaccard index (IoU) | Precision | Recall | F1 score | Accuracy |
| -------------- | ------ | ------------- | ------------------- | --------- | ------ | -------- | -------- |
| version_0 | <ul><li>Sentinel-2 image</li></ul> | Jaccard loss | 0.7682 | 0.8964 | 0.8430 | 0.8689 | 0.9607 |
| **version_1** | **<ul><li>Sentinel-2 image</li><li>PUGS binary mask derived from OSM</li></ul>** | **Jaccard loss** | **0.7855** | 0.91 | 0.8518 | **0.88** | 0.9641 |
| version_2 | <ul><li>Sentinel-2 image</li><li>SDT raster</li></ul> | Jaccard loss | 0.7580 | 0.8865 | 0.8419 | 0.8636 | 0.9590 |
| version_3 | <ul><li>Sentinel-2 image</li></ul> | BCE | 0.7541 | 0.8850 | 0.8361 | 0.8598 | 0.9579 |
| version_4 | <ul><li>Sentinel-2 image</li><li>PUGS binary mask derived from OSM</li></ul> | BCE | 0.7678 | 0.9171 | 0.8251 | 0.8687 | 0.9615 |
| version_5 | <ul><li>Sentinel-2 image</li><li>SDT raster</li></ul> | BCE | 0.7514 | 0.8970 | 0.8224 | 0.8580 | 0.9580 |
| version_6 | <ul><li>Sentinel-2 image</li></ul> | Focal loss | 0.7191 | 0.8585 | 0.8158 | 0.8366 | 0.9508 |
| version_7 | <ul><li>Sentinel-2 image</li><li>PUGS binary mask derived from OSM</li></ul> | Focal loss | 0.7452 | 0.8909 | 0.8201 | 0.8540 | 0.9567 |
| version_8 | <ul><li>Sentinel-2 image</li><li>SDT raster</li></ul> | Focal loss | 0.7251 | 0.8794 | 0.8052 | 0.8406 | 0.9529 |

*BCE = Binary Cross Entropy*

Based on the results, the **best-performing model** uses Sentinel-2 imagery together with a PUGS binary mask derived from OSM as an input and trained with Jaccard loss function.

### Example of the results
These are examples of model prediction outputs from the best-performing model among all experiments.

![result_viz_1](/reports/figures/gt_vs_pred/v1/region_512_1152.png)
![result_viz_2](/reports/figures/gt_vs_pred/v1/region_768_1152.png)

For the full details of model prediction output, please visit [results folder](/results/).

### Further analysis from the model prediction

In the result analysis, the two models are selected based on the highest performance model in baseline (using only Sentinel-2 image) and having additional data from OSM to do the result analysis. These are two selected models:
| Version folder | Input datasets | Loss function |
| -------------- | -------------- | ------------- |
| version_0 | <ul><li>Sentinel-2 image</li></ul> | Jaccard loss |
| version_1 | <ul><li>Sentinel-2 image</li><li>PUGS binary mask derived from OSM</li></ul> | Jaccard loss |

:mag: **Key Findings**
- Model performance improves as green space size increases.
- The model that uses both Sentinel-2 imagery and PUGS binary mask from OSM as input outperforms the one using only Sentinel-2 imagery.
- Regional parks are easiest to detect and both models achieve approximately 99% of recall.
- Small PUGS (e.g., pocket and neighbourhood parks) are more difficult to detect and have lower recall values.
- Regional parks dominate total green space area but make up only a small fraction of the total number of PUGS.
- In contrast, small PUGS has the higheest numbers in terms of PUGS count but contribute little to total area.
- The most significant performance gap between two models gain from using additional data from OSM is seen in small PUGS, especially pocket parks.
- **Conclusion**: Incorporating additional data from OSM helps improve detection of small PUGS.

| Type of PUGS | Size (ha) |
| ------------ | --------- |
| Pocket park | < 0.4 |
| Neighbourhood park | 0.4 – 3 |
| Community park | 3 - 10 |
| Urban park | 10 – 80 |
| Regional park | > 80 |

*Note: PUGS size categories adapted from (Byrne & Sipe, 2010; Choi et al., 2020; Şenik & Uzun, 2022).*

![result_analysis](/reports/figures/others/pugs_size_analysis.png)

To validate outputs, please visit [Reproducibility document](/reproducibility.md)

## License
The license information is divided into three sections:
1. This repository is released under [MIT License](/LICENSE).
2. The data used in this project is under various licenses. Please visit [Data license section](/data/README.md#data-license) to see more detail about license of data used in this project.
3. The model weights, prediction output, and all figures are licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/legalcode.en).

## Contact
For questions or issues, please open an issue on GitHub or contact m.p.likitpanjamanon@student.utwente.nl

## References
- Byrne, J., & Sipe, N. (2010). Green and open space planning for urban consolidation—A review of the literature and best practice. Urban Research Program. https://research-repository.griffith.edu.au/server/api/core/bitstreams/60289e60-4b96-5c4b-99de-d39d2c8db305/content
- Choi, D., Park, K., & Rigolon, A. (2020). From XS to XL Urban Nature: Examining Access to Different Types of Green Space Using a ‘Just Sustainabilities’ Framework. Sustainability, 12(17), Article 17. https://doi.org/10.3390/su12176998
- Şenik, B., & Uzun, O. (2022). A process approach to the open green space system planning. Landscape and Ecological Engineering, 18(2), 203–219. https://doi.org/10.1007/s11355-021-00492-5
