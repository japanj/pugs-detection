# Public Urban Green Spaces Detection

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Public Urban Green Spaces Detection Workflow

**Note: this project is currently under development phase**

## Notebook viewer
| Notebook name                     | Link to view notebook |
| --------------------------------- | --------------------- |
| 2_data_processing_osm.ipynb       | https://nbviewer.org/github/japanj/pugs-detection/blob/dev/notebooks/2_data_processing_osm.ipynb |
| 3_ground_truth_exploration.ipynb  | https://nbviewer.org/github/japanj/pugs-detection/blob/dev/notebooks/3_ground_truth_exploration.ipynb |

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── processed      <- The intermediate data from raw data processing.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `_` delimited description, e.g.
│                         `1_data_acquisition_osm`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         public_urban_green_spaces_detection and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── pugs_detection     <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes public_urban_green_spaces_detection a Python module
    │
    ├── utils.py                <- Store functions used within data processing step
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

