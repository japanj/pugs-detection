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
from rasterstats import zonal_stats

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
    elif gdf.crs.to_epsg() != crs:
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

def set_all_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    seed_everything(42, workers=True)


############# Specific functions for ground truth exploration #############
def find_non_overlap_area(gdf1, gdf2, threshold=0.5):
    """
    Find polygons in gdf1 that don't overlap significantly (>threshold) with any polygon in gdf2.
    """
    gdf1 = gdf1.copy()
    gdf2 = gdf2.copy()

    gdf1 = gdf1.reset_index(drop=True)
    gdf2 = gdf2.reset_index(drop=True)

    # Create a spatial index for faster processing
    sindex = gdf2.sindex
    
    # Track which polygons to keep
    indices_to_exclude = set()

    for row1 in gdf1.itertuples():
        geom1 = row1.geometry
        area1 = geom1.area
        idx1 = row1.Index
        temp_overlap_area_2 = 0
        
        # Skip empty or invalid geometries
        if not geom1.is_valid or geom1.is_empty:
            indices_to_exclude.add(idx1)
            continue
        
        # Find potential intersections using spatial index
        possible_matches_idx = list(sindex.intersection(geom1.bounds))
        possible_matches = gdf2.iloc[possible_matches_idx]
        
        # Check all potential matches
        for row2 in possible_matches.itertuples():
            geom2 = row2.geometry

            # Skip if no intersection
            if not geom1.intersects(geom2):
                continue
                
            # Calculate intersection
            intersection = geom1.intersection(geom2)
            intersection_area = intersection.area
            
            # Calculate overlap percentage
            overlap_pct1 = intersection_area / area1
            
            temp_overlap_area_2 += intersection_area

            # If any overlap is greater than threshold, exclude this polygon
            if overlap_pct1 >= threshold:
                indices_to_exclude.add(idx1)
                break
            
        if temp_overlap_area_2/area1 >= threshold:
            indices_to_exclude.add(idx1)
            
    # Keep polygons that were not excluded
    indices_to_keep = [i for i in range(len(gdf1)) if i not in indices_to_exclude]
    
    if not indices_to_keep:
        return gpd.GeoDataFrame(geometry=[], crs=gdf1.crs)
    
    # Create result GeoDataFrame with the polygons to keep
    result_gdf = gdf1.loc[indices_to_keep].copy()

    return result_gdf

def find_non_overlap_dataset(gdf1, gdf2, gdf3, threshold=0.5):
    """
    Find polygons in gdf1 that don't overlap significantly (>threshold) with any polygon in gdf2 or gdf3.
    """
    # Find non-overlapping polygons with respect to gdf2
    non_overlap_gdf2 = find_non_overlap_area(gdf1, gdf2, threshold)
    
    # Find non-overlapping polygons with respect to gdf3
    non_overlap_gdf3 = find_non_overlap_area(gdf1, gdf3, threshold)

    non_overlap_dataset = gdf1[gdf1['geometry'].isin(non_overlap_gdf2['geometry']) & gdf1['geometry'].isin(non_overlap_gdf3['geometry'])].copy()
    non_overlap_dataset = non_overlap_dataset.reset_index(drop=True)
    
    return non_overlap_dataset

def get_non_overlap_area_insights(non_overlap_area_gdf, groupby_columns):
    # Calculate non-overlapping areas of green and open spaces
    total_area = non_overlap_area_gdf.area.sum()

    # Group by category, sum areas, and calculate percentages
    result_gdf = (
        non_overlap_area_gdf
        .assign(area_m2=non_overlap_area_gdf.area)  # Create area column
        .groupby(groupby_columns)
        .agg({
            'area_m2': 'sum',  # Sum the areas
            'geometry': 'count'  # Count records per category
        })
        .rename(columns={'geometry': 'count'})
        .reset_index()
        .assign(
            area_pct=lambda df: round((df.area_m2/total_area) * 100, 2)  # Calculate percentage
        )
        .sort_values(by='area_pct', ascending=False)
    ).reset_index(drop=True)

    # Prepare columns for largest polygon info
    result_gdf['largest_poly_area'] = 0.0
    result_gdf['largest_poly_id'] = None
    result_gdf['largest_poly_geometry'] = None
    
    # Find largest polygon for each group and add to results
    for i, row in result_gdf.iterrows():
        # Create filter for this group
        if len(groupby_columns) == 1:
            group_filter = non_overlap_area_gdf[groupby_columns[0]] == row[groupby_columns[0]]
        else:
            # For multiple columns
            group_filter = None
            for col in groupby_columns:
                if group_filter is None:
                    group_filter = non_overlap_area_gdf[col] == row[col]
                else:
                    group_filter = group_filter & (non_overlap_area_gdf[col] == row[col])
        
        group_data = non_overlap_area_gdf[group_filter]
        
        if len(group_data) > 0:
            largest_idx = group_data.area.idxmax()
            largest_poly = non_overlap_area_gdf.loc[largest_idx]
            
            # Add largest polygon info to results
            result_gdf.at[i, 'largest_poly_area'] = largest_poly.geometry.area
            result_gdf.at[i, 'largest_poly_id'] = largest_idx
            result_gdf.at[i, 'largest_poly_geometry'] = largest_poly.geometry
    
    # Convert back to GeoDataFrame with the largest polygon geometries
    geo_result = gpd.GeoDataFrame(
        result_gdf, 
        geometry='largest_poly_geometry',
        crs=non_overlap_area_gdf.crs
    )
    
    # Add percentage of largest polygon compared to total area for this group
    # geo_result['largest_poly_pct'] = round((geo_result['largest_poly_area'] / geo_result['area_m2']) * 100, 2)
    
    return geo_result

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

############# Specific functions for new ground truth exploration #############
def find_overlap_area(gdf1, gdf2):
    overlap_area_gdf = gdf1.sjoin(gdf2, how="left", predicate="intersects")
    overlap_area_gdf.loc[~overlap_area_gdf["index_right"].isna(), "overlap"] = "yes"
    overlap_area_gdf.loc[overlap_area_gdf["index_right"].isna(), "overlap"] = "no"

    # Add area columns if they don't exist
    if "area" not in overlap_area_gdf.columns:
        overlap_area_gdf["area"] = overlap_area_gdf.geometry.area

    # Calculate overlap percentage for rows with overlap = 'yes'
    overlap_area_gdf["overlap_pct"] = 0.0

    # Only process rows with overlap
    overlapping_rows = overlap_area_gdf[overlap_area_gdf["overlap"] == "yes"]

    # Process each overlapping row
    for row in overlapping_rows.itertuples():
        raw_overlap_area = row.geometry.intersection(
            gdf2.loc[row.index_right].geometry
        ).area
        overlap_area = (raw_overlap_area / row.geometry.area) * 100
        # print(f"Overlap area: {overlap_area}")
        overlap_area_gdf.loc[
            (overlap_area_gdf["index_right"] == row.index_right)
            & (overlap_area_gdf["geometry"] == row.geometry),
            "overlap_pct",
        ] = overlap_area
        overlap_area_gdf.loc[
            (overlap_area_gdf["index_right"] == row.index_right)
            & (overlap_area_gdf["geometry"] == row.geometry),
            "raw_overlap_area",
        ] = raw_overlap_area

    return overlap_area_gdf

from shapely.ops import unary_union

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