# Data folder

This folder contains all input data used by the workflow and output data from the workflow.

## Folder structure
```
├── README.md           <- README file for data dolfer
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

### raw folder
| Folder/File | Source | Description | Dataset URL |
|-------------|--------|-------------|-------------|
| boundary/<br>dresden_boundary.geojson | OSM | Dresden administrative boundary | - |
| ground truth/<br>DE009L2_DRESDEN_UA2018_v013.gpkg | Copernicus Land Monitoring Service | Land cover and land use data in Functional Urban Areas (FUA) in Europe (The downloaded data is specific for Dresden) | https://land.copernicus.eu/en/products/urban-atlas/urban-atlas-2018 |
| ground truth/<br>green_and_openspaces_dataset.geojson | Dresden Open Data Portal (Origin from Office for Geodata and Cadastre) | Green and Open spaces in Dresden | https://opendata.dresden.de/informationsportal/?open=1&result=271FE127E21B4EB4A39DBB845AE31379#app/mainpage |
| ground truth/<br>leisure_area_dataset.geojson | Dresden Open Data Portal (Origin from Environment Agency) | Dresden administrative boundary | Recreational and leisure areas in Dresden| https://opendata.dresden.de/informationsportal/?open=1&result=56BB16A5A3564732AFAAD3D1524AB1FC#app/mainpage |
| ground truth/<br>park_an_der_zwirnmuhle_polygon.geojson | Author | boundary of Park Zwirnmühle (Manually digitized by Author) | - |
| ground truth/<br>parks_and_greenspaces_dataset.geojson | Dresden Open Data Portal (Origin from Office for Urban Green Spaces and Waste Management) | Park and green spaces in Dresden | https://opendata.dresden.de/informationsportal/?open=1&result=37039CDA0D4E4F3F8AABDDD155F180C6#app/mainpage |
| ground truth/<br>point_parks_and_gardens_dataset.geojson | Dresden Open Data Portal (Origin from Environment Agency) | Location of parks (point dataset) | https://opendata.dresden.de/informationsportal/?open=1&result=EB3A548B330E4C7F963D63FA8EF54DD8#app/mainpage |
| osm/amenity | OSM | POI (e.g. waste baskets, playgrounds, and benches) | - |
| osm/green space | OSM | LULC polygons related to green space | - |
| osm/network | OSM | footpath network | - |
| sentinel-2/<br>cm_sentinel2_l1c_openeo_med_4months_dresden.geotiff | Copernicus Data Hub | Sentinel-2 image (SENTINEL2_L1C colletion) (Applying cloud mask and create median composite on the fly) | - |

### processed folder
| Folder/File | Source | Description | Original data | Modification from original dataset |
|-------------|--------|-------------|---------------|---------------------------|
| `ground truth/i_parks_type_dataset.geojson`| Output from `2_data_processing_ground_truth.ipynb` | Parks (I type) dataset | `leisure_area_dataset.geojson` | Filter only areas that are in *I-Parkanlagen, Zoo, Botanischer Garten* category |
| 1. `ground truth/clipped_dresden_pugs_gt.geojson` <br> 2. `ground truth/clipped_dresden_pugs_gt.geotiff` <br> 3. `ground truth/dresden_pugs_gt.geotiff` | Output from `4_ground_truth_creation.ipynb` | Ground truth of Public Urban Green Spaces (PUGS) | 1. `green_and_openspaces_dataset.geojson` <br> 2. `i_parks_type_dataset.geojson` <br> 3. `parks_and_greenspaces_dataset.geojson` <br> 4. `park_an_der_zwirnmuhle_polygon.geojson` | Filter and merge all four original datasets |
| `ground truth/eua_pugs_dataset.geojson`| Output from `2_data_processing_ground_truth.ipynb` | Reclassified Urban Atlas Land Cover/Land use 2018 (EUA dataset) | `ground truth/DE009L2_DRESDEN_UA2018_v013.gpkg`  | Reclassify the dataset |
| All data in `osm` folder | Output from `2_data_processing_osm.ipynb` | PUGS derived from OSM data | all data in `raw/osm/` folder | Use OSM tag, POI, and footpath network to classify green space (polygons downloaded from OSM) as PUGS or non-PUGS |
| All data in `sentinel-2` folder | Output from `2_data_processing_satellite_image.ipynb` | Sentinel-2 image stack with NDVI raster, and binary mask and Signed-distance transform (SDT) raster derived from OSM data | all data in `raw/sentinel-2/` and `processed/osm` folders | 1. Normalize Sentinel-2 pixel value to range 0 to 1 <br> 2. Calculate NDVI indices <br> 3. Stack NDVI and binary mask and SDT raster derived from OSM |
| All data in `tiles` folder | Output from `5_model_training.ipynb` | Image tiles for training, test, and validation set | `stacked_sentinel2_dresden_clipped.geotiff` | Create image tiles from the processed satellite image |

## Data license

There are four main licenses that applied to the data in this project.
1. Copernicus's data policy
    
   Sentinel-2 data and Urban Atlas Land Cover/Land use 2018 (EUA dataset) are under Copernicus's data policy.

2. Open Data Commons Open Database License (ODbL)

   All OSM data are under ODbL.

3. Datenlizenz Deutschland – Namensnennung – Version 2.0 (dl-by-de/2.0)

   All data downoaded from Dresden Open Data Portal are under dl-by-de/2.0.

4. Creative Commons Attribution 4.0 International (CC-BY-4.0)

   All derivative data (all data in `processed` folder) and data made by author are under CC-BY-4.0.
