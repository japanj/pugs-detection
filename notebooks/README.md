# notebooks folder

This folder contains Jupyter notebooks for each step of the Public Urban Green Spaces (PUGS) detection workflow.

## Workflow Diagram
A diagram shows the recommended execution order and dependencies between the notebooks.
```mermaid
flowchart TD
    A[1_data_acquisition_ground_truth] --> D[2_data_processing_ground_truth]
    B[1_data_acquisition_osm] --> E[2_data_processing_osm]
    C[1_data_acquisition_satellite_image] --> F[3_data_processing_satellite_image]
    D --> G[4_ground_truth_exploration]
    G --> H[5_ground_truth_creation]
    H --> I[6_model_training]
    I --> J[7_model_evaluation]
    E --> F
    F --> I
    J --> K[8_result_analysis]
```

## How to execute the notebooks
- Follow the notebook numbering for the recommended execution order (naming convention of notebook is `<step_number>_<short_description>.ipynb`)
- Data Acquisition notebooks can be skipped and user can use the provided raw data to get the reproducible result in data processing till model evaluation result step.

## Data Locations
- Input and output data for these notebooks are stored in the top-level `data` folder.
- For details on the all data used in this project, see [Data folder document](../data/README.md).

## Notebook List
| Notebook name         |  Description           |
| --------------------- |  --------------------- |
| 1_data_acquisition_ground_truth.ipynb | Download ground truth datasets |
| 1_data_acquisition_osm.ipynb | Download the OpenStreetMap (OSM) data |
| 1_data_acquisition_satellite_image.ipynb | Download Sentinel-2 image |
| 2_data_processing_ground_truth.ipynb | Pre-process the ground truth datasets before ground truth exploration and creation steps |
| 2_data_processing_osm.ipynb | Derive public urban green space (PUGS) from OSM data |
| 3_data_processing_satellite_image.ipynb | Pre-process Sentinel-2 image and stack additional raster derived from OSM data with Sentinel-2 image |
| 4_ground_truth_exploration.ipynb | Explore all ground truth dataset to make a decision on which ground truth datasets should be used to create single ground truth dataset |
| 5_ground_truth_creation.ipynb | Create single ground truth dataset |
| 6_model_training.ipynb | Train the model |
| 7_model_evaluation.ipynb | Evaluate the model performance and save the output from model prediction |
| 8_result_analysis.ipynb | Further prediction result analysis |

## Notebook-specific instructions
- `6_model_training.ipynb`: To change model parameters (e.g., learning rate, batch size) as listed in [Model experiment setup](/models/README.md#model-experiment-setup), edit the **Set variables section** in the notebook. 

    >Model training step can take up **2-3 hours** to complete.

- `7_model_evaluation.ipynb`: To evaluate different model checkpoints, change the model checkpoint path and version folder in the **Set variables section** in the notebook. More details about the model checkpoint files and model experiment setup can be found in [Model folder](/models/README.md).

## Note
- In the data acquisition notebooks, some data sources require users to have an account in order to download data and some dataset require users to download them manually. All details are listed inside the notebooks.
- To run the workflow on **Windows or MacOS**, you need to modify some part of code in `6_model_training.ipynb` and `7_model_evaluation.ipynb`. See [manual modification step](../reproducibility.md#code-modification) for more detail. 