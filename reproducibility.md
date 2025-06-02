# Reproduce the workflow

This document provides additional information for reproducing the workflow and verifying output consistency.

## Set the parameters

Set `os` parameter in [config.yaml](/config.yaml) based on your OS (linux, windows, and macos)

**Note:** When `os: "windows"` or `os: "macos"` is set, `training_num_workers`, `validation_num_workers`, and `test_num_workers` parameters in the config file will **not be applied** to avoid known issues with different start method of multiprocessing on Windows/macOS  and Linux. The DataLoader will use the default single-process data loading instead and it will **not guarantee** the identical model performance metrics and prediction output.

More details about the issue: https://github.com/microsoft/torchgeo/issues/886#issuecomment-1302537713

## Data for Reproducibility

### Use provided raw data files
- The raw data in `data/raw/` must be used to get the reproducible results
- Different data retrieval date can lead to different raw data since OSM data might change over time and Sentinel-2 image collections are occasionally reprocessed. 

    More details about Sentinel-2 processing baseline: https://sentiwiki.copernicus.eu/web/s2-processing

### Use provided model checkpoints
For **model training** step, you have two options:

1. **Train from scratch**: Run notebook `6_model_training.ipynb`
   - New checkpoints will be saved in `models/trained_models/`

2. **Use pre-trained checkpoints** (recommended for exact reproduction)
   - Pre-trained models are available in `models/trained_models/`
   - Set `version` in `config.yaml` to match the experiment you want to reproduce
   - You can directly run `7_model_evaluation.ipynb` for model evaluation

## Workflow Execution

All notebooks required for this workflow are available in the [notebooks folder](/notebooks/) and should be executed in sequence. It also includes instructions for [parameter configuration](/notebooks/README.md#change-the-parameters) to experiment with different model setups.

Please visit [notebooks documentation](/notebooks/README.md) for the workflow execution detail.

## Validate result

You can verify your results match the reference outputs by comparing:

### 1. Validate Model Metrics
- Compare your metrics in `models/test_result/version_{version}/test_metrics.csv` with reference values
- For quick comparison, reference metrics for all model versions are available in the [Results section](/README.md#results), but please note that the metrics is round up in Results section.

### 2. Validate Result Analysis Chart
- Check the area-based analysis in `reports/figures/others/pugs_size_analysis.png` or [Results section](/README.md#results)

### 3. Validate the distribution of classified PUGS based on different classification strategy
- Check the distribution graph in `reports/figures/others/osm_pugs_classification_method.png`

### 4. Validate the intermediate results
- For intermediate results verification, reference HTML notebooks are available in `reports/notebook html/`

**(Optional) Validate Visual Outputs**
- Check prediction outputs in `results/clipped_prediction/clipped_pred_v{version}.geotiff`
- Check visual examples in `reports/figures/gt_vs_pred/v{version}/`

**Note:** `{version}` need to match your `version` setting in `config.yaml`.

:warning: Difference in hardware and OS **does not guarantee** the **identical** output as stated in [PyTorch document](https://docs.pytorch.org/docs/stable/notes/randomness.html#reproducibility), e.g. model performance metrics, model prediction output, and result analysis output.