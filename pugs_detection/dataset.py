import torchgeo
import torch
import torch.nn as nn
import numpy as np
import copy
from torchgeo.datasets import RasterDataset, VectorDataset
from torchgeo.samplers import GridGeoSampler
from torch.utils.data import Dataset
from pyproj import CRS
from pugs_detection.utils import set_all_seeds

def _process_mask(mask, valid_area, band_count):
    mask = mask.numpy()
    valid_area = valid_area.numpy()
    # change the valid area array position to be Red band (important for vegetation detection so I won't replace it)
    if band_count >= 13:
        valid_area_all = valid_area[0] | valid_area[1] | valid_area[2] | valid_area[3] | valid_area[4] | valid_area[5] | valid_area[6] | valid_area[7] | valid_area[8] | valid_area[9] | valid_area[10] | valid_area[11] | valid_area[12]
    else:
        valid_area_all = valid_area[0] | valid_area[1] | valid_area[2] | valid_area[3] | valid_area[4] | valid_area[5] | valid_area[6] | valid_area[7] | valid_area[8]
    # mask[~valid_area[2]] = 0
    mask[~valid_area_all] = 0
    result = torch.from_numpy(mask)
    return result

# Create a custom filtering function
def _filter_patches(sample, band_count):
    """Filter patches that contain only background or only PUGS"""
    min_value = sample['image'].min()
    valid_area = (sample['image']!=min_value) # create a mask of valid area
    
    sample['mask'] = _process_mask(sample['mask'], valid_area, band_count)
    
    mask = sample['mask'].numpy()
    # Calculate percentage of green space in the mask
    green_percentage = np.mean(mask)
    
    # Filter the patches that has only background or only PUGS out
    return 0 < green_percentage < 1

# Custom dataset class that applies filtering
class FilteredGeoDataset(Dataset):
    # stride is set as 64 for both x and y directions
    def __init__(self, dataset, patch_size=256, stride=64, transform=None, specific_bands=list(range(13)), dataset_type='train'):
        self.dataset = dataset
        self.sampler = GridGeoSampler(dataset, size=(patch_size, patch_size), stride=stride)
        self.transform = transform
        self.bounds = dataset.bounds
        self.specific_bands = specific_bands
        self.band_count = len(specific_bands)
        self.dataset_type = dataset_type
        
        # Compute the valid patches
        self.valid_bboxes = []
        count = 0
        if self.dataset_type == 'train':
            for bbox in self.sampler:
                sample = self.dataset[bbox]
                if _filter_patches(sample, self.band_count):
                    self.valid_bboxes.append(bbox)
                count += 1
                print(count)
            print(f"Found {len(self.valid_bboxes)} valid patches out of {len(self.sampler)} total patches")
        else:
            # For validation and test sets, use all patches
            self.valid_bboxes = list(self.sampler)
            print(f"Using all {len(self.valid_bboxes)} patches for {dataset_type} set")
    
    def __len__(self):
        return len(self.valid_bboxes)
    
    def __getitem__(self, idx):
        sample = self.dataset[self.valid_bboxes[idx]]

        # Select specific bands    
        sample['image'] = sample['image'][self.specific_bands]
        min_value = sample['image'].min()
        valid_area = (sample['image']!=min_value) # create a mask of valid area

        # sample['image'], valid_area = contrast_stretch_patch(sample['image'])
        sample['mask'] = _process_mask(sample['mask'], valid_area, self.band_count)
        
        del sample["crs"]
        del sample["bounds"]

        return sample

# def augmented_condition(sample):
#     # Example: Only augment patches with <20% green space
#     mask = sample['mask'].numpy()
#     green_percentage = np.mean(mask)
#     return green_percentage < 0.2

class AugmentedDataset(Dataset):
    """Dataset wrapper that applies multiple augmentations to increase sample count"""
    def __init__(self, dataset, transform_list=None):
        self.dataset = dataset
        self.transform_list = transform_list
    
    def __len__(self):
        return len(self.dataset) * (len(self.transform_list) + 1)  # Original + augmented versions
    
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
        sample_copy['image'] = sample_copy['image'].squeeze(0)
        sample_copy['mask'] = sample_copy['mask'].squeeze(0)
            
        return sample_copy
    
# class AugmentedDataset(Dataset):
#     """Dataset wrapper that applies multiple augmentations to increase sample count"""
#     def __init__(self, dataset, transform_list=None, augmented_condition_fn=None):
#         self.dataset = dataset
#         self.transform_list = transform_list
#         self.augmented_condition_fn = augmented_condition_fn

#         # Pre-calculate which samples should be augmented
#         self.augmentable_indices = []
#         for i in range(len(dataset)):
#             sample = dataset[i]
#             if self.augmented_condition_fn(sample):
#                 self.augmentable_indices.append(i)
        
#         # Calculate total length
#         self.total_length = len(dataset) + len(self.augmentable_indices) * len(self.transform_list)

#         # Create mapping from idx to (sample_idx, aug_idx)
#         self.idx_mapping = []
#         for i in range(len(dataset)):
#             self.idx_mapping.append((i, -1))  # Original samples
#             if i in self.augmentable_indices:
#                 for j in range(len(self.transform_list)):
#                     self.idx_mapping.append((i, j))  # Augmented versions

#     def __len__(self):
#         return self.total_length
    
#     def __getitem__(self, idx):
#         sample_idx, aug_idx = self.idx_mapping[idx]
#         sample = self.dataset[sample_idx]
        
#         # Return original if not augmented
#         if aug_idx == -1:
#             return sample
        
#         # Apply transformation
#         transform = self.transform_list[aug_idx]
#         sample_copy = copy.deepcopy(sample)
#         sample_copy = transform(sample_copy)
        
#         # Ensure proper shape
#         sample_copy['image'] = sample_copy['image'].squeeze(0)
#         sample_copy['mask'] = sample_copy['mask'].squeeze(0)
            
#         return sample_copy
    
# Create datasets for each split
def create_dataset_split(image_path, label_path, epsg_code, band_list, dataset_type, transform_list):
    set_all_seeds(42)
    image_ds = RasterDataset(paths=image_path, crs=CRS.from_epsg(epsg_code), res=10)
    label_ds = VectorDataset(paths=label_path, crs=CRS.from_epsg(epsg_code), res=10)
    combined_ds = image_ds & label_ds
    filter_ds = FilteredGeoDataset(dataset=combined_ds, stride=128, specific_bands=band_list, dataset_type=dataset_type)
    if dataset_type == 'train':
        # return AugmentedDataset(dataset=filter_ds, transform_list=transform_list, augmented_condition_fn=augmented_condition)
        return AugmentedDataset(dataset=filter_ds, transform_list=transform_list)
    else:
        return filter_ds