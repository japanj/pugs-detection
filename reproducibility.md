# Reproduce the workflow

This document provides additional information for reproducing the workflow and verifying output consistency.

## Set the parameters

Set `os` parameter in [config.yaml](/config.yaml) based on your OS (linux, windows, and macos)

**Note:** When `os: "windows"` or `os: "macos"` is set, `training_num_workers`, `validation_num_workers`, and `test_num_workers` parameters in the config file will **not be applied** to avoid known issues with different start method of multiprocessing on Windows/macOS  and Linux. The DataLoader will use the default single-process data loading instead and it will **not guarantee** the identical model performance metrics and prediction output.

More details about the issue: https://github.com/microsoft/torchgeo/issues/886#issuecomment-1302537713

## Data for Reproducibility

This repository includes all raw data files required to reproduce our results:

### Use Provided Raw Data Files
- The raw data in `data/raw/` must be used to get the reproducible results
- Different data retrieval date can lead to different raw data since OSM data might change over time and Sentinel-2 image collections are occasionally reprocessed. 

    More details about Sentinel-2 processing baseline: https://sentiwiki.copernicus.eu/web/s2-processing

### Skip Data Acquisition Notebooks
To ensure reproducibility, it's recommended to:
- Skip the data acquisition notebooks (`1_data_acquisition_*.ipynb`)
- Start workflow execution from data processing notebooks (`2_data_processing_*.ipynb`)
- Use the pre-downloaded raw data files provided in this repository

## Check result

You can verify your results match the reference outputs by comparing:

- **Model performance metrics**: `models/test_result/version_{version}/test_metrics.csv`
- **Prediction outputs**: 
  - `results/whole_area_prediction/pred_v{version}.geotiff`
  - `results/clipped_prediction/clipped_pred_v{version}.geotiff`
- **Result analysis**: `reports/figures/others/overlap_pct_with_gs_size.png`

**Note:** `{version}` need to match your `version` setting in `config.yaml`.

For intermediate results verification, reference HTML notebooks are available in `reports/notebook html/`.

:warning: Difference in hardware and OS **does not guarantee** the **identical** output as stated in [PyTorch document](https://docs.pytorch.org/docs/stable/notes/randomness.html#reproducibility), e.g. model performance metrics, model prediction output, and result analysis output, but overall it should lead to the same conclusion as shown in [result section](/README.md#results).