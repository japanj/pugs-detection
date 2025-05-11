"""
ground_truth.py

This module contains functions for ground truth datasets exploration,
final ground truth dataset creation of public urban green spaces (PUGS),
and any groun truth dataset processing.

Author: Pitchaporn Likitpanjamanon
Date: 01-05-2025
"""

import geopandas as gpd
import pandas as pd
import rasterio
import requests
from rasterstats import zonal_stats
from shapely.ops import unary_union


def download_ground_truth_data(url, params=None, og_crs=4326, new_crs=32633):
    """
    Download ground truth data from a WFS service or OGC API
    and convert to a GeoDataFrame

    Parameters:
    ----------
    url : str
        URL of the WFS service or OGC API endpoint
    params : dict
        Parameters for the request (e.g., filter, version)
    og_crs : int
        Original CRS of the data
    new_crs : int
        Desired CRS for the output GeoDataFrame

    Returns:
    -------
    gdf : GeoDataFrame
        GeoDataFrame containing the downloaded data
    """
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
        raise


def add_ndvi_to_polygons(gdf, raster_path, ndvi_band_index=0):
    """
    Calculate average NDVI values for polygons in a GeoDataFrame

    Parameters:
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing the polygons
    raster_path : str
        Path to the raster file containing NDVI data
    ndvi_band_index : int
        Band index for NDVI data in the raster

    Returns:
    -------
    result_gdf : GeoDataFrame
        GeoDataFrame with an additional column for NDVI mean values
    """
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
            stats=["mean"],
        )

    # Convert to DataFrame and add to the GeoDataFrame
    ndvi_df = pd.DataFrame(ndvi_stats)

    result_gdf["ndvi_mean"] = ndvi_df["mean"].values.round(2)

    return result_gdf


def find_overlap_area(gdf1, gdf2):
    """
    Find the overlapping area between two GeoDataFrames

    Parameters:
    ----------
    gdf1 : GeoDataFrame
        First GeoDataFrame
    gdf2 : GeoDataFrame
        Second GeoDataFrame

    Returns:
    -------
    overlap_area_gdf : GeoDataFrame
        GeoDataFrame containing the overlapping areas and their percentages
    """
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
    """
    Calculate the overlap percentage between two GeoDataFrames
    and report the overlapping percentage

    Parameters:
    ----------
    gdf1 : GeoDataFrame
        First GeoDataFrame.
    gdf2 : GeoDataFrame
        Second GeoDataFrame.
    """
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


def calculate_category_overlap(gdf1, gdf2, category_column="sst_lv_3_liste"):
    """
    Calculate the overlap percentage for each area type in gdf1

    Parameters:
    ----------
    gdf1 : GeoDataFrame
        First GeoDataFrame with area type information
    gdf2 : GeoDataFrame
        Second GeoDataFrame
    category_column : str
        The area type column name for grouping

    Returns:
    -------
    result_df : DataFrame
        DataFrame containing the overlap statistics for each area type
    """
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
        category_stats.append(
            {
                "category": category,
                "total_area": category_area,
                "overlap_area": intersection_area,
                "total_overlap_pct": overlap_percentage,
            }
        )

    # Convert to DataFrame
    geo_df = pd.DataFrame(category_stats)

    # Merge the statistical data with the geometric overlap data
    result_df = pd.merge(
        geo_df, stats_df, left_on="category", right_on=category_column, how="outer"
    )

    # Clean up merged dataframe
    if category_column != "category":
        result_df = result_df.drop(category_column, axis=1)

    # Round values for display
    result_df["total_overlap_pct"] = result_df["total_overlap_pct"].round(2)

    # Sort by true overlap percentage
    result_df = result_df.sort_values("total_overlap_pct", ascending=False)

    result_df = result_df.drop(
        columns=["count", "total_area", "overlap_area", "min_overlap"]
    )

    return result_df


def check_intersecting_point(point_dataset, polygon_dataset, buffer_distance=100):
    """
    Check the intersection of points with polygons in a GeoDataFrame.

    Parameters:
    ----------
    point_dataset : GeoDataFrame
        GeoDataFrame of point dataset
    polygon_dataset : GeoDataFrame
        GeoDataFrame of area/polygon dataset
    buffer_distance : int
        Buffer distance in meters to create a buffer around points
        to check for intersection with polygons

    Returns:
    -------
    points_joined : GeoDataFrame
        GeoDataFrame containing points that intersect with the polygons
    """
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

    return points_joined


def merge_datasets(gdf1, gdf2):
    """
    Merge two GeoDataFrames

    Parameters:
    ----------
    gdf1 : GeoDataFrame
        First GeoDataFrame
    gdf2 : GeoDataFrame
        Second GeoDataFrame

    Returns:
    -------
    merged_dataset : GeoDataFrame
        Merged GeoDataFrame
    """
    gdf1_copy = gdf1.copy()
    gdf2_copy = gdf2.copy()

    if "area" not in gdf1_copy.columns:
        gdf1_copy["area"] = gdf1_copy.geometry.area
    if "area" not in gdf2_copy.columns:
        gdf2_copy["area"] = gdf2_copy.geometry.area

    merged_dataset = pd.concat([gdf1_copy, gdf2_copy], ignore_index=True)
    merged_dataset = gpd.GeoDataFrame(merged_dataset, geometry="geometry", crs=gdf1.crs)

    return merged_dataset


def print_non_green_space_info(gdf, threshold=0):
    """
    Print information about polygons that has average NDVI values below a threshold
    and print the percentage of Low NDVI area out

    Parameters:
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing the polygons
    threshold : float
        NDVI threshold for filtering polygons
    """
    print(f"Area of polygons with NDVI <= {threshold}:")
    low_ndvi_area = gdf[gdf["ndvi_mean"] <= threshold].area.sum()
    total_area = gdf.area.sum()
    area_percentage = (low_ndvi_area / total_area) * 100 if total_area > 0 else 0
    print(
        f"Low NDVI value area: {low_ndvi_area:.2f} m² from {total_area:.2f} m² ({area_percentage:.2f}%)"
    )


def merge_overlapping_polygons(gdf, threshold=0.5):
    """
    Merge (dissolve) overlapping polygons in a GeoDataFrame based on a specified overlap threshold.

    Parameters:
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing the polygons to be merged
    threshold : float
        Overlap threshold for merging polygons (0-1)

    Returns:
    -------
    merged_gdf : GeoDataFrame
        GeoDataFrame containing the merged polygons
    """
    # Use spatial index for efficient overlap search
    gdf = gdf.reset_index(drop=True)
    sindex = gdf.sindex
    merged = []
    id_poly1 = []
    id_poly2_merged = []
    list_merged_id = set()
    for i, poly1 in gdf.iterrows():
        group = [poly1.geometry]
        id_poly2 = []
        possible_matches_index = list(sindex.intersection(poly1.geometry.bounds))
        for j in possible_matches_index:
            if i==j:
                continue
            poly2 = gdf.iloc[j]
            inter = poly1.geometry.intersection(poly2.geometry)
            poly1_area = poly1.geometry.area
            poly2_area = poly2.geometry.area
            if not inter.is_empty:
                overlap_pct_poly1 = inter.area / poly1_area
                overlap_pct_poly2 = inter.area / poly2_area
                if (overlap_pct_poly1 > threshold) or (overlap_pct_poly2 > threshold):
                    id_poly2.append(poly2.unique_id)
                    list_merged_id.add(poly2.unique_id)
                    group.append(poly2.geometry)
        merged.append(unary_union(group))
        id_poly1.append(poly1.unique_id)
        id_poly2_merged.append(id_poly2)
    merged_gdf = gpd.GeoDataFrame(geometry=merged, crs=gdf.crs)
    merged_gdf["id_poly1"] = id_poly1
    merged_gdf["id_poly2"] = id_poly2_merged

    return merged_gdf

def merge_overlapping_polygons_improved(gdf, threshold=0.5):
    """
    Merge overlapping polygons while avoiding duplication by using connected components.
    """
    import networkx as nx
    from shapely.ops import unary_union
    
    # Reset index for consistent indexing
    gdf = gdf.reset_index(drop=True)
    
    # Create spatial index
    sindex = gdf.sindex
    
    # Create a graph where nodes are polygon indices and edges represent significant overlaps
    G = nx.Graph()
    G.add_nodes_from(range(len(gdf)))
    
    # Find overlapping polygons and add edges
    for i, poly1 in gdf.iterrows():
        possible_matches_index = list(sindex.intersection(poly1.geometry.bounds))
        for j in possible_matches_index:
            if i >= j:  # Only check each pair once
                continue
            poly2 = gdf.iloc[j]
            inter = poly1.geometry.intersection(poly2.geometry)
            if not inter.is_empty:
                poly1_area = poly1.geometry.area
                poly2_area = poly2.geometry.area
                overlap_pct_poly1 = inter.area / poly1_area
                overlap_pct_poly2 = inter.area / poly2_area
                if (overlap_pct_poly1 > threshold) or (overlap_pct_poly2 > threshold):
                    G.add_edge(i, j)
    
    # Find connected components (clusters of overlapping polygons)
    connected_components = list(nx.connected_components(G))
    
    # Merge polygons in each connected component
    merged_geometries = []
    component_info = []
    
    for component in connected_components:
        component = list(component)
        geometries = [gdf.loc[i, "geometry"] for i in component]
        merged_geometries.append(unary_union(geometries))
        
        # Store IDs of polygons in this component
        if "unique_id" in gdf.columns:
            primary_id = gdf.loc[component[0], "unique_id"]
            other_ids = [gdf.loc[i, "unique_id"] for i in component[1:]]
        else:
            primary_id = component[0]
            other_ids = component[1:]
            
        component_info.append({"id_poly1": primary_id, "id_poly2": other_ids})
    
    # Create GeoDataFrame with the merged geometries
    result = gpd.GeoDataFrame(component_info, geometry=merged_geometries, crs=gdf.crs)
    
    return result