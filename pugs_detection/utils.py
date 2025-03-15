"""
utils.py

This module contains utility functions for processing geospatial data,
including OSM data and Sentinel-2 image processing.

Author: Pitchaporn Likitpanjamanon
Date: [YYYY-MM-DD]
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
import torch
from lightning.pytorch import seed_everything
from IPython.display import display
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt

def load_osm_data(file_path, crs):
    """
    Load OSM data from a file and convert it to a GeoDataFrame.

    Parameters
    ----------
    file_path : str
        The path to the OSM data file.
    crs : str
        The coordinate reference system (CRS) of the data.

    Returns
    -------
    gdf : GeoDataFrame
        The OSM data as a GeoDataFrame.
    """
    # Load the OSM data
    gdf = gpd.read_file(file_path)

    # Set the CRS
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=crs)
    else:
        gdf = gdf.to_crs(epsg=crs)

    return gdf

def print_basic_info(gdf):
    """
    Print basic information about the OSM data.

    Parameters
    ----------
    gdf : GeoDataFrame
        The OSM data as a GeoDataFrame.
    """
    # Print the basic information
    print("Number of rows:", len(gdf))
    print("Number of columns:", len(gdf.columns))
    display(gdf.head())

def print_classification_report(gdf):
    """
    Print a classification report for the OSM data.

    Parameters
    ----------
    gdf : GeoDataFrame
        The OSM data as a GeoDataFrame.
    """
    total_green_area = gdf['area_left'].sum()

    # Print the classification report
    print("total LULC spaces:", len(gdf['id_left']))
    print("total labeled LULC spaces:", len(gdf[~gdf['is_public'].isna()]['id_left']))
    print("total public LULC spaces:", len(gdf[gdf['is_public']=='yes']['id_left']))
    print("total non-public LULC spaces:", len(gdf[gdf['is_public']=='no']['id_left']))
    print("-"*20)
    print("total LULC spaces (area):", total_green_area)
    print("percentage of labeled LULC spaces (area):", (gdf[~gdf['is_public'].isna()]['area_left'].sum()/total_green_area)*100)
    print("percentage of public LULC spaces (area):", (gdf[gdf['is_public']=='yes']['area_left'].sum()/total_green_area)*100)
    print("percentage of non-public LULC spaces (area):", (gdf[gdf['is_public']=='no']['area_left'].sum()/total_green_area)*100)

def calculate_poi_number(lulc_gdf, poi_gdf, osm_key):
    """
    Calculate the number of points of interest (POIs) within each LULC space.

    Parameters
    ----------
    lulc_gdf : GeoDataFrame
        The LULC data as a GeoDataFrame.
    poi_gdf : GeoDataFrame
        The POI data as a GeoDataFrame.
    osm_key : str
        The OSM key to use for grouping the POIs.
    
    Returns
    -------
    lulc_gdf : GeoDataFrame
        The LULC data with the number of POIs within each LULC space.
    """
    # Avoid to change the original df
    modified_lulc_gdf = lulc_gdf.copy()
    # Drop duplicate so we can get the real number of POIs
    lulc_gdf_wo_duplicate = lulc_gdf.drop_duplicates(subset=['id_left'])

    poi_lulc_gdf = poi_gdf.sjoin(lulc_gdf_wo_duplicate, how='inner', predicate='within', rsuffix='_right2')
    grouped_poi_gdf = poi_lulc_gdf.groupby(by=['id_left', osm_key]).size().reset_index(name='count')
    grouped_poi_gdf = pd.pivot_table(grouped_poi_gdf, values='count', index='id_left', columns=osm_key, fill_value=0)
    column_names = grouped_poi_gdf.columns.tolist()
    modified_lulc_gdf = modified_lulc_gdf.merge(grouped_poi_gdf, how='left', left_on='id_left', right_on='id_left')
    modified_lulc_gdf[column_names] = modified_lulc_gdf[column_names].fillna(0)

    return modified_lulc_gdf

def calculate_footpath_length(lulc_gdf, footpath_gdf):
    # Avoid to change the original df
    modified_lulc_gdf = lulc_gdf.copy()
    # Drop duplicate so we can get the real number of POIs
    lulc_gdf_wo_duplicate = lulc_gdf.drop_duplicates(subset=['id_left'])

    lulc_gdf_with_footpath = lulc_gdf_wo_duplicate.sjoin(footpath_gdf, how='left', predicate='intersects', rsuffix='_right2')
    footpath_length_sum_gdf = lulc_gdf_with_footpath.groupby(by='id_left')['length'].sum().reset_index()
    modified_lulc_gdf = modified_lulc_gdf.merge(footpath_length_sum_gdf, how='left', left_on='id_left', right_on='id_left')

    return modified_lulc_gdf

# _ indicates internal use only (user shouldn't call it directly)
def _create_estimated_area_dictionary(lulc_gdf):
    leisure_node_element_list = lulc_gdf[(lulc_gdf['element_right']=='node')&
                                         (lulc_gdf['id_left']!=lulc_gdf['id_right'])]['leisure_right'].unique().tolist()
    landuse_node_element_list = lulc_gdf[(lulc_gdf['element_right']=='node')&
                                         (lulc_gdf['id_left']!=lulc_gdf['id_right'])]['landuse_right'].unique().tolist()
    print("leisure tag of node element in polygon:", leisure_node_element_list)
    print("landuse tag of node element in polygon:", landuse_node_element_list)

    space_with_polygon = lulc_gdf[((lulc_gdf['element_right']=='way')|
                                   (lulc_gdf['element_right']=='relation'))&
                                   (lulc_gdf['id_left']!=lulc_gdf['id_right'])]

    pc_area_dict = {}
    for row in space_with_polygon.itertuples():
        if (row.leisure_right != None) & (row.leisure_right in leisure_node_element_list):
            pc_area = (row.area_right/row.area_left)*100
            if row.leisure_right not in pc_area_dict:
                pc_area_dict[row.leisure_right] = []
                pc_area_dict[row.leisure_right].append(pc_area)
            else:
                pc_area_dict[row.leisure_right].append(pc_area)
        if (row.landuse_right != None)&(row.landuse_right in landuse_node_element_list):
            pc_area = (row.area_right/row.area_left)*100
            if row.landuse_right not in pc_area_dict:
                pc_area_dict[row.landuse_right] = []
                pc_area_dict[row.landuse_right].append(pc_area)
            else:
                pc_area_dict[row.landuse_right].append(pc_area)

    pc_avg_area = {k : round(np.mean(v)) for k, v in pc_area_dict.items()}
    print("estimated average percentage of each leisure/landuse:", pc_avg_area)

    # If there is no estimated area of specific leisure/landuse, set it to 1
    for i in leisure_node_element_list:
        if i not in pc_avg_area:
            pc_avg_area[i] = 1
    for i in landuse_node_element_list:
        if i not in pc_avg_area:
            pc_avg_area[i] = 1

    print("final estimated average percentage area:", pc_avg_area)

    return pc_avg_area

def classification_with_smaller_area(lulc_gdf):
    # Avoid to change the original df
    modified_lulc_gdf = lulc_gdf.copy()

    pc_avg_area = _create_estimated_area_dictionary(modified_lulc_gdf)
    # Need to check the access tag of the small polygon -> so get unique big polygon first
    # Then, get access of small polygon
    unlabel_green_space = modified_lulc_gdf[modified_lulc_gdf['is_public'].isna()]
    big_polygon_id = unlabel_green_space['id_left'].unique().tolist()
    check_big_polygon = [] # Big polygon that is assigned as non-public from this step

    for i in big_polygon_id:
        small_polygon_id = modified_lulc_gdf[(modified_lulc_gdf['id_left']==i)&(modified_lulc_gdf['id_left']!=modified_lulc_gdf['id_right'])]['id_right'].unique().tolist()
        pub_area = 0
        nonpub_area = 0
        total_area = modified_lulc_gdf[modified_lulc_gdf['id_left']==i]['area_left'].values[0]
        for j in small_polygon_id:
            # Check 'is_public' column of small polygon and save its area
            temp = modified_lulc_gdf[modified_lulc_gdf['id_left']==j]
            if temp['is_public'].values[0]=='yes':
                if temp['element_left'].values[0]=='node':
                    if temp['leisure_left'].values[0] != None:
                        pc = pc_avg_area[temp['leisure_left'].values[0]]
                        pub_area += (pc*total_area)/100
                    elif temp['landuse_left'].values[0] != None:
                        pc = pc_avg_area[temp['landuse_left'].values[0]]
                        pub_area += (pc*total_area)/100
                else:
                    pub_area += temp['area_left'].values[0]
            else:
                if temp['element_left'].values[0]=='node':
                    if temp['leisure_left'].values[0] != None:
                        pc = pc_avg_area[temp['leisure_right'].values[0]]
                        nonpub_area += (pc*total_area)/100
                    elif temp['landuse_left'].values[0] != None:
                        pc = pc_avg_area[temp['landuse_left'].values[0]]
                        nonpub_area += (pc*total_area)/100
                else:
                    nonpub_area += temp['area_left'].values[0]
            # count += 1
        # after applying threshold, only 1 area is classified since some small polygon inside is unclassified.
        if (pub_area > nonpub_area) & (pub_area>=((50*total_area)/100)):
            # print('yes')
            modified_lulc_gdf.loc[modified_lulc_gdf['id_left']==i,'is_public'] = 'yes'
        elif (pub_area < nonpub_area) & (nonpub_area>=((50*total_area)/100)):
            # print('no')
            check_big_polygon.append(i)
            print("larger polygon id:", i, "   small polygon id:", small_polygon_id) 
            modified_lulc_gdf.loc[modified_lulc_gdf['id_left']==i,'is_public'] = 'no'
        
    return modified_lulc_gdf
        
def create_binary_mask(gdf, file_path, satellite_image_path):
    with rasterio.open(satellite_image_path) as src:
        transform = src.transform
        width = src.width
        height = src.height
        crs = src.crs

        # Create an empty raster with the same properties as the satellite image
        out_shape = (height, width)
        raster = np.zeros(out_shape, dtype='uint8')

        # Rasterize the polygons
        rasterized = rasterize(
            [(geom, 1) for geom in gdf.geometry],
            out_shape=out_shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype='uint8'
        )
        
        # Save the raster to a file
        with rasterio.open(
            file_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=raster.dtype,
            crs=gdf.crs,
            transform=transform,
        ) as dst:
            dst.write(rasterized, 1)

    print("Rasterization complete.")

def create_sdt_raster(gdf, file_path, satellite_image_path):
    with rasterio.open(satellite_image_path) as src:
        transform = src.transform
        width = src.width
        height = src.height

        # Create an empty raster with the same properties as the satellite image
        out_shape = (height, width)
        raster = np.zeros(out_shape, dtype='uint8')
        
        # Rasterize the polygons
        raster = rasterize(
            [(geom, 1) for geom in gdf.geometry],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            all_touched=True,
            dtype='uint8'
        )

        # Compute the distance transform for the inside and outside of the polygons
        distance_inside = distance_transform_edt(raster) # Calculate the nearest distance from each pixel inside polygon to the nearest pixel outside polygon
        distance_outside = distance_transform_edt(1 - raster) # Calculate the nearest distance from each pixel outside polygon to the nearest pixel inside polygon

        # Create the signed distance transform
        signed_distance_transform = distance_inside - distance_outside

        # Save the signed distance transform to a file
        with rasterio.open(
            file_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=signed_distance_transform.dtype,
            crs=gdf.crs,
            transform=transform,
        ) as dst:
            dst.write(signed_distance_transform, 1)

        print("Signed distance transform complete.")

def contrast_stretch(array, lower_percentile=1, upper_percentile=99):
    # Apply a percentile-based contrast stretch (ignore the extreme values so image isn't too dark or too bright)
    lower = np.percentile(array, lower_percentile)
    upper = np.percentile(array, upper_percentile)
    array = np.clip(array, lower, upper) # clip -> limit the values in an array to specific range
    return (array - lower) / (upper - lower)

# Min-max scaling to [-1, 1] range
def normalize_sdt_minmax(sdt_array):
    sdt_min = float(sdt_array.min())
    sdt_max = float(sdt_array.max())
    # Apply min-max formula scaled to [-1, 1]
    normalized = 2 * (sdt_array - sdt_min) / (sdt_max - sdt_min) - 1
    return normalized

def set_all_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    seed_everything(42, workers=True)