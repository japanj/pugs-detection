"""
dataset.py

This module contains classes to create datasets for training, validation, and testing
and functions to create the dataset.

Author: Pitchaporn Likitpanjamanon
Date: 01-05-2025
"""

import torch
import numpy as np
import copy
import rasterio
from tqdm import tqdm
from torchgeo.datasets import RasterDataset, VectorDataset
from torchgeo.samplers import GridGeoSampler
from torch.utils.data import Dataset
from pyproj import CRS
from pugs_detection.utils import set_all_seeds
from torch.utils.data import Dataset
from rasterio.windows import Window


def _process_mask(mask, valid_area, band_count):
    """
    Process the mask to remove invalid areas by setting those areas to 0.

    Parameters:
    -----------
    mask : torch.Tensor
        The mask tensor to be processed.
    valid_area : torch.Tensor
        The valid area tensor indicating valid pixels.
    band_count : int
        The number of bands in the dataset.

    Returns:
    --------
    result: torch.Tensor
        The processed mask with invalid areas set to 0.
    """
    mask = mask.numpy()
    valid_area = valid_area.numpy()
    if band_count >= 13:
        valid_area_all = (
            valid_area[0]
            | valid_area[1]
            | valid_area[2]
            | valid_area[3]
            | valid_area[4]
            | valid_area[5]
            | valid_area[6]
            | valid_area[7]
            | valid_area[8]
            | valid_area[9]
            | valid_area[10]
            | valid_area[11]
            | valid_area[12]
        )
    else:
        valid_area_all = (
            valid_area[0]
            | valid_area[1]
            | valid_area[2]
            | valid_area[3]
            | valid_area[4]
            | valid_area[5]
            | valid_area[6]
            | valid_area[7]
            | valid_area[8]
        )
    # mask[~valid_area[2]] = 0
    mask[~valid_area_all] = 0
    result = torch.from_numpy(mask)
    return result


def _filter_patches(sample, band_count):
    """
    Filter patches that contain only background or only PUGS out
    to keep the dataset balanced

    Parameters:
    -----------
    sample : dict
        The sample dictionary containing image and mask data.
    band_count : int
        The number of bands in the dataset.

    Returns:
    --------
    bool
        True if the patch contains both background and PUGS. Otherwise, return False.
    """
    # min_value = sample['image'].min()
    # valid_area = (sample['image']!=min_value) # create a mask of valid area

    nodata_value = -9999
    valid_area = sample["image"] != nodata_value

    sample["mask"] = _process_mask(sample["mask"], valid_area, band_count)

    mask = sample["mask"].numpy()
    # Calculate percentage of green space in the mask
    green_percentage = np.mean(mask)

    # Filter the patches that has only background out
    return 0 < green_percentage


class FilteredGeoDataset(Dataset):
    """
    Dataset wrapper that filters patches based on the presence of PUGS
    and background. It uses a grid sampling approach to create patches
    from the input dataset.

    Parameters:
    -----------
    dataset : IntersectionDataset
        The input dataset containing the images and masks.
    patch_size : int
        The size of the patches to be created.
    stride : int
        The stride used for sampling patches.
    transform : callable
        A function/transform to apply to the sample.
    specific_bands : list
        List of specific bands to be used from the dataset.
    dataset_type : str
        The type of dataset ('train', 'validation', 'test').
    """

    def __init__(
        self,
        dataset,
        patch_size=256,
        stride=64,
        transform=None,
        specific_bands=list(range(13)),
        dataset_type="train",
    ):
        self.dataset = dataset
        self.sampler = GridGeoSampler(
            dataset, size=(patch_size, patch_size), stride=stride
        )
        self.transform = transform
        self.bounds = dataset.bounds
        self.specific_bands = specific_bands
        self.band_count = len(specific_bands)
        self.dataset_type = dataset_type

        # Compute the valid patches
        self.valid_bboxes = []

        # Get total number of patches for progress bar
        total_patches = len(self.sampler)

        if self.dataset_type == "train":
            for bbox in tqdm(
                self.sampler,
                desc=f"Filtering patches for {dataset_type}",
                total=total_patches,
                unit="patch",
            ):
                sample = self.dataset[bbox]
                if _filter_patches(sample, self.band_count):
                    self.valid_bboxes.append(bbox)
            print(
                f"Found {len(self.valid_bboxes)} valid patches out of {len(self.sampler)} total patches"
            )
        else:
            # For validation and test sets, use all patches
            self.valid_bboxes = list(self.sampler)
            print(f"Using all {len(self.valid_bboxes)} patches for {dataset_type} set")

    def __len__(self):
        return len(self.valid_bboxes)

    def __getitem__(self, idx):
        sample = self.dataset[self.valid_bboxes[idx]]

        # Select specific bands
        sample["image"] = sample["image"][self.specific_bands]
        # min_value = sample['image'].min()
        nodata_value = -9999
        valid_area = sample["image"] != nodata_value  # create a mask of valid area

        # sample['image'], valid_area = contrast_stretch_patch(sample['image'])
        sample["mask"] = _process_mask(sample["mask"], valid_area, self.band_count)

        # replace nodata values with 0
        sample["image"][sample["image"] == nodata_value] = 0

        del sample["crs"]
        del sample["bounds"]

        return sample


class AugmentedDataset(Dataset):
    """
    Dataset wrapper that applies multiple augmentations to the original dataset.

    Parameters:
    -----------
    dataset : FilteredGeoDataset
        The original dataset containing images and masks.
    transform_list : list
        List of transformations to apply to the dataset.
    """

    def __init__(self, dataset, transform_list=None):
        self.dataset = dataset
        self.transform_list = transform_list

    def __len__(self):
        return len(self.dataset) * (
            len(self.transform_list) + 1
        )  # Original + augmented versions

    def __getitem__(self, idx):
        # Calculate original dataset index and augmentation version
        dataset_idx = idx // (len(self.transform_list) + 1)
        aug_version = idx % (len(self.transform_list) + 1)

        # Get original sample
        sample = self.dataset[dataset_idx]

        # If it's the first version (aug_version=0), return original
        if aug_version == 0 or self.transform_list is None:
            return sample

        # Apply transformation
        transform = self.transform_list[aug_version - 1]

        # Make a deep copy to avoid modifying the original
        sample_copy = copy.deepcopy(sample)

        # Apply the transformation
        sample_copy = transform(sample_copy)

        # Ensure proper shape (remove batch dimension added by some transforms)
        sample_copy["image"] = sample_copy["image"].squeeze(0)
        sample_copy["mask"] = sample_copy["mask"].squeeze(0)

        return sample_copy


class PredictedImageDataset(Dataset):
    """
    Dataset for the whole area prediction using a trained model.

    Parameters:
    -----------
    image_path : str
        Path to the input image.
    patch_size : int
        Size of the patches to be created.
    stride : int
        Stride used for sampling patches.
    band_list_predict : list
        List of the image's bands/channels.
    """

    def __init__(
        self,
        image_path,
        patch_size=256,
        stride=256,
        band_list_predict=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    ):
        self.image_path = image_path
        self.patch_size = patch_size
        self.stride = stride
        self.band_list_predict = band_list_predict

        # Open the image to get dimensions
        with rasterio.open(image_path) as src:
            self.height = src.height
            self.width = src.width
            self.count = src.count
            self.transform = src.transform
            self.crs = src.crs

        # Create list of windows - all using Window objects
        self.windows = []
        for y in range(0, self.height - patch_size + 1, stride):
            for x in range(0, self.width - patch_size + 1, stride):
                self.windows.append(Window(x, y, patch_size, patch_size))

        # Edge patches along bottom
        if self.height % stride != 0:
            last_y = self.height - patch_size
            for x in range(0, self.width - patch_size + 1, stride):
                self.windows.append(Window(x, last_y, patch_size, patch_size))

        # Edge patches along right side
        if self.width % stride != 0:
            last_x = self.width - patch_size
            for y in range(0, self.height - patch_size + 1, stride):
                self.windows.append(Window(last_x, y, patch_size, patch_size))

        # Corner patch (if needed)
        if self.width % stride != 0 and self.height % stride != 0:
            self.windows.append(
                Window(
                    self.width - patch_size,
                    self.height - patch_size,
                    patch_size,
                    patch_size,
                )
            )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]

        with rasterio.open(self.image_path) as src:
            # Read image data for this window
            image = src.read(window=window)
            # image = image[0:14, :, :]  # Select specific bands
            image = image[self.band_list_predict]  # Select specific bands

        return {
            "image": image,
            "window_info": window,  # Store coordinates as tuple
        }


def create_dataset_split(
    image_path, label_path, epsg_code, band_list, dataset_type, transform_list, seed=42
):
    """
    Create a dataset for training, validation, or testing.

    Parameters:
    -----------
    image_path : str
        Path to the input image.
    label_path : str
        Path to the PUGS ground truth vector file.
    epsg_code : int
        EPSG code for the coordinate reference system.
    band_list : list
        List of the image's bands/channels.
    dataset_type : str
        Type of dataset ('train', 'validation', 'test').
    transform_list : list
        List of transformations to apply to the dataset.
    seed : int
        Random seed for reproducibility.

    Returns:
    --------
    Dataset (either AugmentedDataset or FilteredGeoDataset depending on dataset_type)
        The created dataset to be used for training, validation, or testing
    """
    set_all_seeds(seed)
    image_ds = RasterDataset(paths=image_path, crs=CRS.from_epsg(epsg_code), res=10)
    label_ds = VectorDataset(paths=label_path, crs=CRS.from_epsg(epsg_code), res=10)
    combined_ds = image_ds & label_ds
    filter_ds = FilteredGeoDataset(
        dataset=combined_ds,
        stride=128,
        specific_bands=band_list,
        dataset_type=dataset_type,
    )
    if dataset_type == "train":
        # return AugmentedDataset(dataset=filter_ds, transform_list=transform_list, augmented_condition_fn=augmented_condition)
        return AugmentedDataset(dataset=filter_ds, transform_list=transform_list)
    else:
        return filter_ds
