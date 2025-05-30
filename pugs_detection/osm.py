"""
osm.py

This module contains functions for loading and processing OpenStreetMap (OSM) data.

Author: Pitchaporn Likitpanjamanon
Date: 01-05-2025
"""

import geopandas as gpd
import pandas as pd
import numpy as np


def load_osm_data(file_path, crs):
    """
    Load OSM data from a file and convert it to a GeoDataFrame.

    Parameters:
    -----------
    file_path : str
        The path to the OSM data file.
    crs : int
        The coordinate reference system (CRS) of the data.

    Returns:
    --------
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


def print_classification_report(gdf):
    """
    Print a classification report for the OSM data.

    Parameters:
    -----------
    gdf : GeoDataFrame
        GeoDataFrame of the OSM data
    """
    total_green_area = gdf["area_left"].sum()

    # Print the classification report
    print(f"total LULC spaces: {len(gdf['id_left'])}")
    print(f"total labeled LULC spaces: {len(gdf[~gdf['is_public'].isna()]['id_left'])}")
    print(f"total public LULC spaces: {len(gdf[gdf['is_public'] == 'yes']['id_left'])}")
    print(
        f"total non-public LULC spaces: {len(gdf[gdf['is_public'] == 'no']['id_left'])}"
    )
    print("-" * 20)
    print(f"total LULC spaces (area): {total_green_area}")
    print(
        f"percentage of labeled LULC spaces (area): {round((gdf[~gdf['is_public'].isna()]['area_left'].sum() / total_green_area) * 100, 2)}%"
    )
    print(
        f"percentage of public LULC spaces (area): {round((gdf[gdf['is_public'] == 'yes']['area_left'].sum() / total_green_area) * 100, 2)}%"
    )
    print(
        f"percentage of non-public LULC spaces (area): {round((gdf[gdf['is_public'] == 'no']['area_left'].sum() / total_green_area) * 100, 2)}%"
    )


def calculate_poi_number(lulc_gdf, poi_gdf, osm_key):
    """
    Calculate the number of points of interest (POIs) within each LULC space.

    Parameters:
    -----------
    lulc_gdf : GeoDataFrame
        The LULC data as a GeoDataFrame.
    poi_gdf : GeoDataFrame
        The POI data as a GeoDataFrame.
    osm_key : str
        The OSM key to use for grouping the POIs.

    Returns:
    --------
    lulc_gdf : GeoDataFrame
        The LULC data with the number of POIs within each LULC space.
    """
    # Avoid to change the original df
    modified_lulc_gdf = lulc_gdf.copy()
    # Drop duplicate so we can get the real number of POIs
    lulc_gdf_wo_duplicate = lulc_gdf.drop_duplicates(subset=["id_left"])

    poi_lulc_gdf = poi_gdf.sjoin(
        lulc_gdf_wo_duplicate, how="inner", predicate="within", rsuffix="_right2"
    )
    grouped_poi_gdf = (
        poi_lulc_gdf.groupby(by=["id_left", osm_key]).size().reset_index(name="count")
    )
    grouped_poi_gdf = pd.pivot_table(
        grouped_poi_gdf, values="count", index="id_left", columns=osm_key, fill_value=0
    )
    column_names = grouped_poi_gdf.columns.tolist()
    modified_lulc_gdf = modified_lulc_gdf.merge(
        grouped_poi_gdf, how="left", left_on="id_left", right_on="id_left"
    )
    modified_lulc_gdf[column_names] = modified_lulc_gdf[column_names].fillna(0)

    return modified_lulc_gdf


def calculate_footpath_length(lulc_gdf, footpath_gdf):
    """
    Calculate the total length of footpaths within each LULC space

    Parameters:
    -----------
    lulc_gdf : GeoDataFrame
        GeoDataFrame of the LULC data
    footpath_gdf : GeoDataFrame
        GeoDataFrame of the footpath data

    Returns:
    --------
    modified_lulc_gdf : GeoDataFrame
        The LULC data with the total length of footpaths within each LULC space
    """
    modified_lulc_gdf = lulc_gdf.copy()
    # Drop duplicate so we can get the real number of POIs
    lulc_gdf_wo_duplicate = lulc_gdf.drop_duplicates(subset=["id_left"])

    lulc_gdf_with_footpath = lulc_gdf_wo_duplicate.sjoin(
        footpath_gdf, how="left", predicate="intersects", rsuffix="_right2"
    )
    footpath_length_sum_gdf = (
        lulc_gdf_with_footpath.groupby(by="id_left")["length"].sum().reset_index()
    )
    modified_lulc_gdf = modified_lulc_gdf.merge(
        footpath_length_sum_gdf, how="left", left_on="id_left", right_on="id_left"
    )

    return modified_lulc_gdf


# _ indicates internal use only (user shouldn't call it directly)
def _create_estimated_area_dictionary(lulc_gdf):
    """
    Create a dictionary of estimated area for each leisure/landuse tag
    based on the area of the polygon

    Parameters:
    -----------
    lulc_gdf : GeoDataFrame
        GeoDataFrame of the LULC data

    Returns:
    --------
    pc_avg_area : dict
        Dictionary of estimated area for each leisure/landuse tag
    """
    leisure_node_element_list = (
        lulc_gdf[
            (lulc_gdf["element_right"] == "node")
            & (lulc_gdf["id_left"] != lulc_gdf["id_right"])
        ]["leisure_right"]
        .unique()
        .tolist()
    )
    landuse_node_element_list = (
        lulc_gdf[
            (lulc_gdf["element_right"] == "node")
            & (lulc_gdf["id_left"] != lulc_gdf["id_right"])
        ]["landuse_right"]
        .unique()
        .tolist()
    )
    print("leisure tag of node element in polygon:", leisure_node_element_list)
    print("landuse tag of node element in polygon:", landuse_node_element_list)

    space_with_polygon = lulc_gdf[
        (
            (lulc_gdf["element_right"] == "way")
            | (lulc_gdf["element_right"] == "relation")
        )
        & (lulc_gdf["id_left"] != lulc_gdf["id_right"])
    ]

    pc_area_dict = {}
    for row in space_with_polygon.itertuples():
        if (row.leisure_right != None) & (
            row.leisure_right in leisure_node_element_list
        ):
            pc_area = (row.area_right / row.area_left) * 100
            if row.leisure_right not in pc_area_dict:
                pc_area_dict[row.leisure_right] = []
                pc_area_dict[row.leisure_right].append(pc_area)
            else:
                pc_area_dict[row.leisure_right].append(pc_area)
        if (row.landuse_right != None) & (
            row.landuse_right in landuse_node_element_list
        ):
            pc_area = (row.area_right / row.area_left) * 100
            if row.landuse_right not in pc_area_dict:
                pc_area_dict[row.landuse_right] = []
                pc_area_dict[row.landuse_right].append(pc_area)
            else:
                pc_area_dict[row.landuse_right].append(pc_area)

    pc_avg_area = {k: round(np.mean(v)) for k, v in pc_area_dict.items()}
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
    """
    Use the smaller area of the polygon to classify the larger polygon
    based on the percentage of public and non-public areas
    (If 50% of the smaller area is public, classify the larger polygon as public)

    Parameters:
    -----------
    lulc_gdf : GeoDataFrame
        GeoDataFrame of the LULC data

    Returns:
    --------
    modified_lulc_gdf : GeoDataFrame
        The LULC data with the access classification based on the smaller area
    """
    modified_lulc_gdf = lulc_gdf.copy()

    pc_avg_area = _create_estimated_area_dictionary(modified_lulc_gdf)
    # Need to check the access tag of the small polygon -> so get unique big polygon first
    # Then, get access of small polygon
    unlabel_green_space = modified_lulc_gdf[modified_lulc_gdf["is_public"].isna()]
    big_polygon_id = unlabel_green_space["id_left"].unique().tolist()
    check_big_polygon = []  # Big polygon that is assigned as non-public from this step

    for i in big_polygon_id:
        small_polygon_id = (
            modified_lulc_gdf[
                (modified_lulc_gdf["id_left"] == i)
                & (modified_lulc_gdf["id_left"] != modified_lulc_gdf["id_right"])
            ]["id_right"]
            .unique()
            .tolist()
        )
        pub_area = 0
        nonpub_area = 0
        total_area = modified_lulc_gdf[modified_lulc_gdf["id_left"] == i][
            "area_left"
        ].values[0]
        for j in small_polygon_id:
            # Check 'is_public' column of small polygon and save its area
            temp = modified_lulc_gdf[modified_lulc_gdf["id_left"] == j]
            if temp["is_public"].values[0] == "yes":
                if temp["element_left"].values[0] == "node":
                    if temp["leisure_left"].values[0] != None:
                        pc = pc_avg_area[temp["leisure_left"].values[0]]
                        pub_area += (pc * total_area) / 100
                    elif temp["landuse_left"].values[0] != None:
                        pc = pc_avg_area[temp["landuse_left"].values[0]]
                        pub_area += (pc * total_area) / 100
                else:
                    pub_area += temp["area_left"].values[0]
            else:
                if temp["element_left"].values[0] == "node":
                    if temp["leisure_left"].values[0] != None:
                        pc = pc_avg_area[temp["leisure_right"].values[0]]
                        nonpub_area += (pc * total_area) / 100
                    elif temp["landuse_left"].values[0] != None:
                        pc = pc_avg_area[temp["landuse_left"].values[0]]
                        nonpub_area += (pc * total_area) / 100
                else:
                    nonpub_area += temp["area_left"].values[0]
            # count += 1
        # after applying threshold, only 1 area is classified since some small polygon inside is unclassified.
        if (pub_area > nonpub_area) & (pub_area >= ((50 * total_area) / 100)):
            # print('yes')
            modified_lulc_gdf.loc[modified_lulc_gdf["id_left"] == i, ["is_public", "classification_step"]] = ["yes", "hierachy"]
        elif (pub_area < nonpub_area) & (nonpub_area >= ((50 * total_area) / 100)):
            # print('no')
            check_big_polygon.append(i)
            print("larger polygon id:", i, "   small polygon id:", small_polygon_id)
            modified_lulc_gdf.loc[modified_lulc_gdf["id_left"] == i, ["is_public", "classification_step"]] = ["no", "hierachy"]

    return modified_lulc_gdf
