# Models folder

This folder contains various model checkpoints and default model performance metrics (accuracy, Jaccard index, loss) from Torchgeo library.

It contains two sub-folders: `checkpoints` and `test_result`

## `checkpoints` folder

This folder contains sub-folders of different training experiments (`version_<number>`). Inside sub-folder, it contains:
- `checkpoints` folder: contains the best two models of each training experiment
- `hparams.yaml` file: contains the hyperparameters setup of each training experiment <br> *Note: num_channels is always 13 because it doesn't take a log of first layer customization into account*
- `metrics.csv` file: contains training and validation metrics (accuracy, Jaccard index, loss) recorded during model training

## `test_result` folder

This folder contains default model performance metris (accuracy, Jaccard index, loss) on test set.

Each folder (`version_<number>`) contains the model performance on test set of each experiment.

## Model experiment setup

| Version folder | input  | Number of input channels | Loss function | 
| -------------- | ------ | ------------------------ | ------------- |
| version_0 | 1. Sentinel-2 image | 13 channels | Jaccard loss | 
| version_1 | 1. Sentinel-2 image <br> 2. PUGS binary mask derived from OSM | 14 channels | Jaccard loss |
| version_2 | 1. Sentinel-2 image <br> 2. SDT raster | 14 channels | Jaccard loss |
| version_3 | 1. Sentinel-2 image | 13 channels | BCE |
| version_4 | 1. Sentinel-2 image <br> 2. PUGS binary mask derived from OSM | 14 channels | BCE |
| version_5 | 1. Sentinel-2 image <br> 2. SDT raster | 14 channels | BCE |
| version_6 | 1. Sentinel-2 image | 13 channels | Focal loss |
| version_7 | 1. Sentinel-2 image <br> 2. PUGS binary mask derived from OSM | 14 channels | Focal loss |
| version_8 | 1. Sentinel-2 image <br> 2. SDT raster | 14 channels | Focal loss |

**Hyperparameters setup**
- Epochs = 50
- Learning rate = 0.0001
- Pre-trained weight (for Sentinel-2 image) = [ResNet50_Weights.SENTINEL2_ALL_MOCO](https://torchgeo.readthedocs.io/en/stable/api/models.html#sentinel-2)
  Note: This weight is available in Torchgeo library and originally from https://github.com/zhu-xlab/SSL4EO-S12 and it is licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/legalcode.en).
- Optimizer = AdamW
- Early stopping: patience = 10
