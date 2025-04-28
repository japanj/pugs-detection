"""
ground_truth.py

This module contains functions for ground truth datasets exploration and 
final ground truth dataset creation of public urban green spaces (PUGS).

Author: Pitchaporn Likitpanjamanon
Date: [YYYY-MM-DD]
"""

import geopandas as gpd
import pandas as pd
import rasterio
import requests
from rasterstats import zonal_stats
from shapely.ops import unary_union

def download_ground_truth_data(url, params=None, og_crs=4326, new_crs=32633):
    try:
        # Make the request
        response = requests.get(url, params=params, timeout=60)

        # Check if the request was successful
        response.raise_for_status()

        gdf = gpd.read_file(response.content)
        
        # Convert to a consistent CRS
        if gdf.crs is not None:
            gdf = gdf.to_crs(new_crs)
        else:
            gdf = gdf.set_crs(og_crs)
            gdf = gdf.to_crs(new_crs)
        print(f"Dataset CRS: {gdf.crs}")
        return gdf
    except requests.exceptions.RequestException as e:
        print(f"Error making request to WFS service: {e}")
        return None
    
def add_ndvi_to_polygons(gdf, raster_path, ndvi_band_index=0):
    result_gdf = gdf.copy()
    
    # Open the raster and read the NDVI band
    with rasterio.open(raster_path) as src:
        ndvi_data = src.read(ndvi_band_index)
        
        affine = src.transform
        nodata = src.nodata

        # Calculate zonal statistics using the extracted NDVI band
        ndvi_stats = zonal_stats(
            result_gdf.geometry, 
            ndvi_data,
            affine=affine,
            nodata=nodata,
            stats=['mean'],
        )
    
    # Convert to DataFrame and add to the GeoDataFrame
    ndvi_df = pd.DataFrame(ndvi_stats)
    
    result_gdf['ndvi_mean'] = ndvi_df['mean'].values.round(2)
    
    return result_gdf

def find_overlap_area(gdf1, gdf2):
    overlap_area_gdf = gdf1.sjoin(gdf2, how="left", predicate="intersects")
    overlap_area_gdf.loc[~overlap_area_gdf["index_right"].isna(), "overlap"] = "yes"
    overlap_area_gdf.loc[overlap_area_gdf["index_right"].isna(), "overlap"] = "no"

    # Add area columns if they don't exist
    if "area" not in overlap_area_gdf.columns:
        overlap_area_gdf["area"] = overlap_area_gdf.geometry.area

    # Calculate overlap percentage for rows with overlap = 'yes'
    overlap_area_gdf["overlap_pct"] = 0.0
    overlap_area_gdf["raw_overlap_area"] = 0.0

    overlap_area_gdf = overlap_area_gdf.reset_index(drop=True)
    # Get indices of overlapping rows
    overlap_idx = overlap_area_gdf[overlap_area_gdf["overlap"] == "yes"].index
    
    # Process each overlapping row
    for idx in overlap_idx:
        row = overlap_area_gdf.loc[idx]
        geom1 = row.geometry
        geom2 = gdf2.loc[row.index_right].geometry
        
        # Calculate intersection
        raw_overlap_area = geom1.intersection(geom2).area
        overlap_pct = (raw_overlap_area / row.geometry.area) * 100
        
        # Assign values
        overlap_area_gdf.loc[idx, "raw_overlap_area"] = raw_overlap_area
        overlap_area_gdf.loc[idx, "overlap_pct"] = overlap_pct

    return overlap_area_gdf

def calculate_dataset_overlap(gdf1, gdf2):
    gdf1_union = unary_union(gdf1.geometry)
    gdf2_union = unary_union(gdf2.geometry)
    
    gdf1_area = gdf1_union.area
    gdf2_area = gdf2_union.area
    
    intersection = gdf1_union.intersection(gdf2_union)
    intersection_area = intersection.area
    
    gdf1_overlap_pct = (intersection_area / gdf1_area) * 100
    gdf2_overlap_pct = (intersection_area / gdf2_area) * 100
    
    print(f"Dataset 1 overlap percentage: {gdf1_overlap_pct:.2f}%")
    print(f"Dataset 2 overlap percentage: {gdf2_overlap_pct:.2f}%")

def calculate_category_overlap(gdf1, gdf2, category_column='sst_lv_3_liste'):
    stats_df = (
        gdf1[gdf1["overlap"] == "yes"]
        .groupby(category_column)
        .agg(
            count=("overlap_pct", "size"),
            min_overlap=("overlap_pct", "min"),
            max_overlap=("overlap_pct", "max"),
            avg_overlap=("overlap_pct", "mean"),
        )
        .reset_index()
    )
    
    # Round the statistical calculations
    stats_df["min_overlap"] = stats_df["min_overlap"].round(2)
    stats_df["max_overlap"] = stats_df["max_overlap"].round(2)
    stats_df["avg_overlap"] = stats_df["avg_overlap"].round(2)

    # Create a union of all geometries in gdf2
    gdf2_union = unary_union(gdf2.geometry)
    
    # Prepare results
    category_stats = []
    
    # For each category value
    for category in gdf1[category_column].unique():
        # Extract all polygons of this category
        category_polygons = gdf1[gdf1[category_column] == category]
        
        # Skip if empty
        if len(category_polygons) == 0:
            continue
        
        # Create a union of all geometries in this category
        category_union = unary_union(category_polygons.geometry)
        category_area = category_union.area
        
        # Calculate intersection with gdf2
        intersection = category_union.intersection(gdf2_union)
        intersection_area = intersection.area
        
        # Calculate percentage
        overlap_percentage = (intersection_area / category_area) * 100
        
        # Store results
        category_stats.append({
            'category': category,
            'total_area': category_area,
            'overlap_area': intersection_area,
            'total_overlap_pct': overlap_percentage,
        })
    
    # Convert to DataFrame
    geo_df = pd.DataFrame(category_stats)
    
    # Merge the statistical data with the geometric overlap data
    result_df = pd.merge(
        geo_df,
        stats_df,
        left_on='category',
        right_on=category_column,
        how='outer'
    )
    
    # Clean up merged dataframe
    if category_column != 'category':
        result_df = result_df.drop(category_column, axis=1)
    
    # Round values for display
    result_df['total_overlap_pct'] = result_df['total_overlap_pct'].round(2)
    
    # Sort by true overlap percentage
    result_df = result_df.sort_values('total_overlap_pct', ascending=False)
    
    result_df = result_df.drop(columns=['count', 'total_area', 'overlap_area', 'min_overlap'])

    return result_df

def check_intersecting_point(point_dataset, polygon_dataset, buffer_distance=100):
    buffer_distance = buffer_distance

    point_park_garden_buffered = point_dataset.copy()
    point_park_garden_buffered["geometry"] = point_dataset.geometry.buffer(
        buffer_distance
    )

    points_joined = point_park_garden_buffered.sjoin(
        polygon_dataset, how="inner", predicate="intersects"
    )

    # Count unique points that intersect
    unique_intersecting_points = points_joined.index.nunique()

    # Get total count of points
    total_points = len(point_park_garden_buffered)

    intersection_percentage = unique_intersecting_points / total_points * 100

    print(
        f"Points intersecting with dataset: {unique_intersecting_points} out of {total_points} ({intersection_percentage:.2f}%)"
    )
    points_joined = points_joined

    return points_joined

def merge_datasets(gdf1, gdf2):
    gdf1_copy = gdf1.copy()
    gdf2_copy = gdf2.copy()

    if "area" not in gdf1_copy.columns:
        gdf1_copy["area"] = gdf1_copy.geometry.area
    if "area" not in gdf2_copy.columns:
        gdf2_copy["area"] = gdf2_copy.geometry.area

    merged_dataset = pd.concat(
        [gdf1_copy, gdf2_copy], ignore_index=True
    )
    merged_dataset = gpd.GeoDataFrame(
        merged_dataset, geometry="geometry", crs=gdf1.crs
    )

    return merged_dataset

def print_non_green_space_info(gdf, threshold=0):
    print(f"Area of polygons with NDVI <= {threshold}:")
    low_ndvi_area = gdf[gdf["ndvi_mean"] <= threshold].area.sum()
    total_area = gdf.area.sum()
    area_percentage = (low_ndvi_area / total_area) * 100 if total_area > 0 else 0
    print(
        f"Low NDVI value area: {low_ndvi_area:.2f} m² from {total_area:.2f} m² ({area_percentage:.2f}%)"
    )