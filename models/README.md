# models folder

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

*PUGS = Public Urban Green Space, OSM = OpenStreetMap, SDT = Signed-distance transform, BCE = Binary Cross Entropy*

**Hyperparameters setup**
- Epochs = 50
- Learning rate = 0.0001
- Pre-trained weight (for Sentinel-2 image) = [ResNet50_Weights.SENTINEL2_ALL_MOCO](https://torchgeo.readthedocs.io/en/stable/api/models.html#sentinel-2) <br>
  *Note: This weight is available in Torchgeo library and originally from https://github.com/zhu-xlab/SSL4EO-S12 and it is licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/legalcode.en).*
- Optimizer = AdamW
- Early stopping: patience = 10

*Note: Weight for additional data from OSM (PUGS binary mask and SDT) is initialized by using Kaiming He initialization (He et al., 2015)*

**Reference:**
- He, K., Zhang, X., Ren, S., & Sun, J. (2015, February 6). Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification. arXiv.org. https://arxiv.org/abs/1502.01852v1

## Model checkpoint path
The table below shows the best model of each experiment. These model checkpoint paths are in `/checkpoints/version_<number>/` folder.
| Version folder | input  | Loss function | The best model (checkpoint path) |
| -------------- | ------ | ------------- | -------------------------------- |
| version_0 | 1. Sentinel-2 image | Jaccard loss | epoch=49-val_loss=0.22.ckpt |
| version_1 | 1. Sentinel-2 image <br> 2. PUGS binary mask derived from OSM | Jaccard loss | epoch=48-val_loss=0.18.ckpt |
| version_2 | 1. Sentinel-2 image <br> 2. SDT raster | Jaccard loss | epoch=48-val_loss=0.22.ckpt |
| version_3 | 1. Sentinel-2 image | BCE | epoch=24-val_loss=0.14.ckpt |
| version_4 | 1. Sentinel-2 image <br> 2. PUGS binary mask derived from OSM | BCE | epoch=28-val_loss=0.13.ckpt |
| version_5 | 1. Sentinel-2 image <br> 2. SDT raster | BCE | epoch=25-val_loss=0.14.ckpt |
| version_6 | 1. Sentinel-2 image | Focal loss | epoch=27-val_loss=0.00.ckpt |
| version_7 | 1. Sentinel-2 image <br> 2. PUGS binary mask derived from OSM | Focal loss | epoch=31-val_loss=0.00.ckpt |
| version_8 | 1. Sentinel-2 image <br> 2. SDT raster | Focal loss | epoch=27-val_loss=0.00.ckpt |