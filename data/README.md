# data folder

This folder contains all input datasets used in the workflow, including both raw data and processed data generated during data preparation steps. It does **not** store model prediction outputs. The model prediction outputs are saved in the top-level [results folder](/results/).

**All the data used in the workflow can be downloaded from https://doi.org/10.5281/zenodo.15553942**

## Folder structure
```
├── README.md           <- README file for data folder
├── processed           <- Intermediate data from original data processing
│   ├── ground truth    <- Processed ground truth datasets
│   ├── osm             <- Processed OpenStreetMap(OSM) data
│   ├── sentinel-2      <- Processed Sentinel-2 data
│   └── tiles           <- Image tiles created from Sentinel-2 image 
│                          for train, test, and validation set creation
│
└── raw                 <- Original data downloaded from data source
    ├── boundary        <- Dresden administrative boundary
    ├── ground truth    <- Ground truth dataset
    ├── osm             <- OSM data (LULC polygons, Point of Interest (POI), footpath network)
    └── sentinel-2      <- Sentinel-2 image
```

## Data description

### `raw` folder

#### Administrative & Boundary Data
| File | Source | Description | Download date |
|------|--------|-------------| ------------- |
| `boundary/ dresden_boundary.geojson` | OSM | Dresden administrative boundary | 30/04/2025 |

#### Ground Truth Data
| File | Source | Description | Dataset URL | Download date | Temporal extent | Last modified date |
|------|--------|-------------|-------------| ------------- | --------------- | ------------------ |
| `ground truth/ DE009L2_DRESDEN_UA2018_v013.gpkg` | Copernicus Land Monitoring Service | Land cover and land use data in Functional Urban Areas (FUA) (This dataset is downloaded by specifying only Dresden area) | [Urban Atlas Land Cover/Land use 2018](https://land.copernicus.eu/en/products/urban-atlas/urban-atlas-2018) | 18/02/2025 | 2017-2019 | - |
| `ground truth/ green_and_openspaces_dataset.geojson` | Dresden Open Data Portal | Green and Open spaces in Dresden | [Grün- und Freiflächen](https://kommisdd.dresden.de/net3/public/ogc.ashx?Service=IKX&NodeId=1638&RenderHint=TargetHTML) <br> (Green and open spaces) | 14/05/2025 | - | 23/05/2024 |
| `ground truth/ leisure_area_dataset.geojson` | Dresden Open Data Portal | Recreational and leisure areas in Dresden | [Nutzungsarten (Strukturtypen) – Kartierung](https://kommisdd.dresden.de/net3/public/ogc.ashx?Service=IKX&NodeId=1059&RenderHint=TargetHTML) <br> (Types of use - structure types - Mapping) | 14/05/2025 | - | - |
| `ground truth/ park_an_der_zwirnmuhle_polygon.geojson` | Author | Boundary of Park Zwirnmühle (Manually digitized) | - | - | - | 29/04/2025 |
| `ground truth/ parks_and_greenspaces_dataset.geojson` | Dresden Open Data Portal | Park and green spaces in Dresden | [Park- und Grünanlagen (Flächen)](https://kommisdd.dresden.de/net3/public/ogc.ashx?Service=IKX&NodeId=751&RenderHint=TargetHTML) <br> (Park and green spaces) | 14/05/2025 | - | 09/11/2020 |
| `ground truth/ point_parks_and_gardens_dataset.geojson` | Dresden Open Data Portal | Location of parks (point dataset) | [Parkanlagen, Gärten](https://kommisdd.dresden.de/net3/public/ogc.ashx?Service=IKX&NodeId=138&RenderHint=TargetHTML) (Parks, gardens) | 14/05/2025 | - | 14/03/2025 |

#### OpenStreetMap (OSM) Data
| File | Source | Description | Download date |
|------|--------|-------------| ------------- |
| `osm/amenity` | OSM | POI (e.g. waste baskets, playgrounds, and benches) | 14/05/2025 |
| `osm/green space` | OSM | LULC polygons related to green space | 14/05/2025 |
| `osm/network` | OSM | footpath network | 14/05/2025 |

#### Satellite Imagery
| File | Source | Description | Download date | Temporal extent |
|------|--------|-------------| ------------- | --------------- |
| `sentinel-2/ cm_sentinel2_l1c_openeo_med_4months_dresden.geotiff` | Copernicus Data Hub | Sentinel-2 image (SENTINEL2_L1C collection) with cloud mask and median composite | 14/05/2025 | 01/05/2024 - 31/08/2024 |

### `processed` folder

#### Ground Truth Derived Data
| Folder/File | Source | Description | Original data | Modification from original dataset |
|-------------|--------|-------------|---------------|------------------------------------|
| `ground truth/ i_parks_type_dataset.geojson`| Output from `2_data_processing_ground_truth.ipynb` | Parks (I type) dataset | `raw/ground truth/ leisure_area_dataset.geojson` | Filter only areas that are in *I-Parkanlagen, Zoo, Botanischer Garten* category |
| 1. `ground truth/ clipped_dresden_pugs_gt.geojson` <br> 2. `ground truth/ clipped_dresden_pugs_gt.geotiff` <br> 3. `ground truth/ dresden_pugs_gt.geotiff` | Output from `4_ground_truth_creation.ipynb` | Ground truth of Public Urban Green Spaces (PUGS) | 1. `raw/ground truth/ green_and_openspaces_dataset.geojson` <br> 2. `raw/ground truth/ i_parks_type_dataset.geojson` <br> 3. `raw/ground truth/ parks_and_greenspaces_dataset.geojson` <br> 4. `raw/ground truth/ park_an_der_zwirnmuhle_polygon.geojson` | Filter and merge all four original datasets |
| `ground truth/ eua_pugs_dataset.geojson`| Output from `2_data_processing_ground_truth.ipynb` | Reclassified Urban Atlas Land Cover/Land use 2018 (EUA dataset) | `raw/ground truth/ DE009L2_DRESDEN_UA2018_v013.gpkg`  | Reclassify the dataset |

#### OSM Processed Data
| Folder/File | Source | Description | Original data | Modification from original dataset |
|-------------|--------|-------------|---------------|------------------------------------|
| All data in `osm` folder | Output from `2_data_processing_osm.ipynb` | PUGS derived from OSM data | all data in `raw/osm/` folder | Use OSM tag, POI, and footpath network to classify green space (polygons downloaded from OSM) as PUGS or non-PUGS |

#### Satellite Image Processing
| Folder/File | Source | Description | Original data | Modification from original dataset |
|-------------|--------|-------------|---------------|------------------------------------|
| All data in `sentinel-2` folder | Output from `2_data_processing _satellite_image.ipynb` | Sentinel-2 image stack with NDVI raster, and binary mask and Signed-distance transform (SDT) raster derived from OSM data | all data in `raw/sentinel-2/` and `processed/osm` folders | 1. Normalize Sentinel-2 pixel value to range 0 to 1 <br> 2. Calculate NDVI indices <br> 3. Stack NDVI and binary mask and SDT raster derived from OSM |

#### Training, test, and validation Data
| Folder/File | Source | Description | Original data | Modification from original dataset |
|-------------|--------|-------------|---------------|------------------------------------|
| All data in `tiles` folder | Output from `5_model_training.ipynb` | Image tiles for training, test, and validation set | `processed/sentinel-2/ stacked_sentinel2_dresden_clipped.geotiff` | Create image tiles from the processed satellite image |

## Data license 

There are four main licenses that applied to the data in this project.
1. **Copernicus's data policy**
    
   Sentinel-2 data and Urban Atlas Land Cover/Land use 2018 (EUA) data are under Copernicus's data policy.<br>
   More details about data policy: [Sentinel-2 data policy](https://sentinel.esa.int/documents/247904/690755/Sentinel_Data_Legal_Notice) and [EUA data policy](https://land.copernicus.eu/en/data-policy)

2. **Open Data Commons Open Database License (ODbL)**

   All OSM data and modification of OSM data are licensed under ODbL.<br>
   More details about license: [ODbL](https://opendatacommons.org/licenses/odbl/1-0/)

3. **Datenlizenz Deutschland – Namensnennung – Version 2.0 (dl-by-de/2.0)**

   All data downloaded from Dresden Open Data Portal are licensed under dl-by-de/2.0.<br>
   More details about license: [dl-by-de/2.0](https://www.govdata.de/dl-de/by-2-0)

4. **Creative Commons Attribution 4.0 International (CC-BY-4.0)**

   All derivative data (all data in `processed` folder) and data made by author are licensed under CC-BY-4.0.<br>
   More details about license: [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/legalcode.en)

## Data Attribution

The following attributions are provided to give credit to the original data sources used in this project:

1. **Original data and derivative works from Sentinel-2 data**

   Contains modified Copernicus Sentinel data [2025].

2. **Original data and derivative works from EUA dataset**

   Generated using European Union's Copernicus Land Monitoring Service information; https://doi.org/10.2909/fb4dffa1-6ceb-4cc0-8372-1ed354c285e6

3. **Original data and derivative works from OpenStreetMap (OSM)**

   Contains information from OpenStreetMap (OSM), which is made available here under the Open Database License (ODbL).

4. **Original data and derivative works from Dresden Open Data Portal**

   Data made available in Dresden Open Data Portal is licensed under dl-de/by-2-0.