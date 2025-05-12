# Reproduce the workflow

## Set up the environment

1. Create the environment with required dependencies for this project
    - **For Linux/WSL:**
        ```
        conda env create -f environment_linux.yml
        ```
    - **For Windows or macOS:**
        ```
        conda env create -f environment_windows_mac.yml
        ```

2. Activate the environment
    ```
    conda activate pugs-detection
    ```

(Optional) User can check all the installed libraries or dependencies by using `conda list` command. 

## Set the parameters

Set `os` parameter in [config.yaml](/config.yaml) based on your OS (linux, windows, and macos)

**Note:** When `os: "windows"` or `os: "macos"` is set, `training_num_workers`, `validation_num_workers`, and `test_num_workers` parameters in the config file will **not be applied** to avoid known issues with different start method of multiprocessing on Windows/macOS  and Linux. The DataLoader will use the default single-process data loading instead and it will **not guarantee** the identical model performance metrics and prediction output.

More details about the issue: https://github.com/microsoft/torchgeo/issues/886#issuecomment-1302537713