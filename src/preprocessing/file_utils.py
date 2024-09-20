"""
This module contains functions for loading satellite data from a directory of
tiles.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import tifffile
import xarray as xr

# local modules
sys.path.append(".")
from src.utilities import SatelliteType


def process_viirs_filename(file_path: Path) -> Tuple[str, str]:
    """
    This function takes in the file_path of a VIIRS file and outputs
    a tuple containing two strings, in the format (date, band). It parses
    the date using the datetime.strptime function with the corresponding
    format: https://docs.python.org/3/library/datetime.html#format-codes
    and then converts it to a string.

    Example input: C:/users/foo/data/DNB_VNP46A1_A2020221.tif
    Example output: ("2020-08-08", "0")

    The format of the date within the filename is {year}{day}, as
    the day here is special, it is the day of the year as a zero-padded
    decimal number like: [001, 002, …, 365] (the format code for this is %j).
    For the example input above, 2020221 is the 221st day of the year 2020 (don't ask
    why they format it like this I got no idea), which translates to August 8th, 2020

    Parameters
    ----------
    file_path : Path
        The Path of the VIIRS file.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the date and band.
    """
    full_date = file_path.name[-11:-4]
    year = full_date[:4]
    day = full_date[4:]
    date = datetime.strptime(f"{year}{day}", "%Y%j").strftime("%Y-%m-%d")
    return (date, "0")




def process_s1_filename(file_path: Path) -> Tuple[str, str]:
    """
    This function takes in the file_path of a Sentinel-1 file and outputs
    a tuple containing two strings, in the format (date, band). It parses
    the date using the datetime.strptime function with the corresponding
    format: https://docs.python.org/3/library/datetime.html#format-codes
    and then converts it to a string.

    Example input: C:/users/foo/data/S1A_IW_GRDH_20200804_VV.tif
    Example output: ("2020-8-4", "VV")

    The format of the date within the filename is {year}{month}{day}.

    Parameters
    ----------
    file_path : Path
        The Path of the Sentinel-1 file.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the date and band.
    """
    splits = file_path.name.split("_")
    full_date = splits[-2]
    year = full_date[:4]
    month = full_date[4:6]
    day = full_date[6:]
    band = splits[-1][:-4]
    date = datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d").strftime("%Y-%m-%d")
    return (date, band)


def process_s2_filename(file_path: Path) -> Tuple[str, str]:
    """
    This function takes in the file_path of a Sentinel-2 file and outputs
    a tuple containing two strings, in the format (date, band). It parses
    the date using the datetime.strptime function with the corresponding
    format: https://docs.python.org/3/library/datetime.html#format-codes
    and then converts it to a string.

    Example input: C:/users/foo/data/L2A_20200816_B01.tif
    Example output: ("2020-8-16", "01")

    The format of the date within the filename is {year}{month}{day}.

    Parameters
    ----------
    file_path : Path
        The Path of the Sentinel-2 file.

    Returns
    -------
    Tuple[str, str]
    """
    splits = file_path.name.split("_")
    full_date = splits[-2]
    year = full_date[:4]
    month = full_date[4:6]
    day = full_date[6:]
    band = splits[-1][1:-4]
    date = datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d").strftime("%Y-%m-%d")
    return (date, band)


def process_landsat_filename(file_path: Path) -> Tuple[str, str]:
    """
    This function takes in the file_path of a Landsat file and outputs
    a tuple containing two strings, in the format (date, band). It parses
    the date using the datetime.strptime function with the corresponding
    format: https://docs.python.org/3/library/datetime.html#format-codes
    and then converts it to a string.

    Example input: C:/users/foo/data/LC08_L1TP_2020-08-30_B9.tif
    Example output: ("2020-8-30", "9")

    The format of the date within the filename is {year}-{month}-{day}.

    Parameters
    ----------
    file_path : Path
        The Path of the Landsat file.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the date and band.
    """
    splits = file_path.name.split("_")
    date = splits[-2]
    band = splits[-1]
    return (date, band[1:band.index(".")])


def process_ground_truth_filename(file_path: Path) -> Tuple[str, str]:
    """
    This function takes in the file_path of the ground truth file and returns
    ("0", "0"), as there is only one ground truth file. The date is set to the
    datetime.min.date() and converted to a string.

    Example input: C:/users/foo/data/groundTruth.tif
    Example output: ("0001-01-01, "0")

    The date for the ground truth will just be the datetime.min.date().

    Parameters
    ----------
    file_path: Path
        The Path of the ground truth file, and though we will ignore it,
        we still need to pass it.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the date and band.
    """
    return (datetime.min.date().isoformat(), "0")


def get_filename_pattern(satellite_type: SatelliteType) -> str:
    """
    Return the file_path pattern for the given satellite type
    using the SatelliteType.value as a key for the patterns dict.

    Parameters
    ----------
    satellite_type : SatelliteType
        The type of satellite being used. SatelliteType is an Enum

    Returns
    -------
    str
        The file_path pattern for the given satellite type.
    """
    patterns = {
        "viirs": "DNB_VNP46A1_",
        "sentinel1": "S1A_IW_GRDH_",
        "sentinel2": "L2A_",
        "landsat": "LC08_L1TP_",
        "gt": "groundTruth.tif",
    }
    return patterns[satellite_type.value]


def get_satellite_files(tile_dir: Path, satellite_type: SatelliteType) -> List[Path]:
    """
    Retrieve all satellite files in the tile directory matching the satellite type pattern using
    the get_filename_pattern function.

    Parameters
    ----------
    tile_dir : Path
        The directory containing the satellite tiles.
    satellite_type : SatelliteType
        The type of satellite, should be one of SatelliteType.VIIRS,
        SatelliteType.S1, SatelliteType.S2, SatelliteType.LANDSAT, or SatelliteType.GT.

    Returns
    -------
    List[Path]
        A list of Path objects for each satellite file that was in the tile directory.
    """
    pattern = get_filename_pattern(satellite_type)
    valid_files = []
    for file in tile_dir.iterdir():
        if pattern in file.name and file.suffix == ".tif" and file.is_file():
            valid_files.append(file)
    return valid_files


def get_grouping_function(satellite_type: SatelliteType) -> Callable:
    """
    Return the function to group satellite files by date and band using the
    SatelliteType.value as a key for the patterns dict.

    Parameters
    ----------
    satellite_type : SatelliteType
        The type of satellite, should be one of SatelliteType.VIIRS,
        SatelliteType.S1, SatelliteType.S2, SatelliteType.LANDSAT, or SatelliteType.GT.

    Returns
    -------
    Callable (function)
        The function to group satellite files by date and band.
    """
    patterns = {
        "viirs": process_viirs_filename,
        "sentinel1": process_s1_filename,
        "sentinel2": process_s2_filename,
        "landsat": process_landsat_filename,
        "gt": process_ground_truth_filename,
    }
    # --- start here ---
    return patterns[satellite_type.value]


def get_unique_dates_and_bands(
    tile_dir: Path, satellite_type: SatelliteType
) -> Tuple[List[str], List[str]]:
    """
    Extracts unique dates and bands from tile directory.
    Generates a set of dates and a set of bands using the grouping function to extract
    the date and band from each file in the satellite files. Those 2 sets are then sorted
    and returned.

    Parameters
    ----------
    tile_dir : Path
        The Tile directory containing the satellite tif files.
    satellite_type : SatelliteType
        The type of satellite, should be one of SatelliteType.VIIRS,
        SatelliteType.S1, SatelliteType.S2, SatelliteType.LANDSAT, or SatelliteType.GT.

    Returns
    -------
    Tuple[List[str], List[str]]
        A tuple containing two lists of the unique dates and bands.
        Each list is a sorted set of the dates and bands.
        The dates list is index 0; The bands list is index 1.
    """
    satellite_files, grouping_function = get_satellite_files(
        tile_dir, satellite_type
    ), get_grouping_function(satellite_type)
    # --- start here ---
    dates = set()
    bands = set()
    for file in satellite_files:
        date, band = grouping_function(file)
        dates.add(date)
        bands.add(band)
    return (sorted(dates), sorted(bands))


def get_parent_tile_id(tile_dir: Path) -> str:
    """
    Returns the name (parent_tile_id) of the tile_dir as a string.

    Parameters
    ----------
    tile_dir : Path
        The Tile directory containing the satellite tif files.

    Returns
    -------
    str
        The parent_tile_id of the path
    """
    return tile_dir.name


def read_satellite_file(satellite_file: Path) -> np.ndarray:
    """
    Reads the satellite file into a np.ndarray with dtype=np.float32 using tifffile.imread.

    Normally, you would use a library like rasterio: https://rasterio.readthedocs.io/en/stable/
    to read the files, but because the dataset does NOT contain any metadata (IEEE has chosen to remove it),
    we can just use tifffile to easily read the data.

    Parameters
    ----------
    satellite_file : Path
        A Path object to the satellite file.

    Returns
    -------
    2 dimensional np.ndarray of shape (height, width)

    """
    file_array = tifffile.imread(satellite_file, dtype=np.float32)
    return file_array


def load_satellite(tile_dir: Path, satellite_type: SatelliteType) -> xr.DataArray:
    """
    Load all bands for a given satellite type from a directory of tile files. Loads the data by
    stacking it into a single 4D np.ndarray of dimensions (band, date, height, width). This is then
    stored in an xarray: https://docs.xarray.dev/en/stable/
    along with the relevant metadata as attributes of the xarray.

    Parameters
    ----------
    tile_dir : Path
        The Tile directory containing the satellite tif files.
    satellite_type : SatelliteType
        The type of satellite, should be one of SatelliteType.VIIRS,
        SatelliteType.S1, SatelliteType.S2, SatelliteType.LANDSAT, or SatelliteType.GT.

    Returns
    -------
    xr.DataArray
        An xr.DataArray containing the satellite data with dimensions (date, band, height, width)
        and the satellite_type, tile_dir, and parent_tile_id attributes.
    """
    # get the files from the tile_dir and the corresponding grouping function
    file_names, grouping_function = get_satellite_files(
        tile_dir, satellite_type
    ), get_grouping_function(satellite_type)
    # get the parent_tile_id
    parent_tile_id = get_parent_tile_id(tile_dir)
    # get the unique dates and bands from the tile_dir
    dates, bands = get_unique_dates_and_bands(tile_dir, satellite_type)

    # --- start here ---
    # create an empty list to store the data along the date dimension
    date_list = []
    # iterate over the dates
    for date in dates:
        # create an empty list to store the data along the band dimension
        band_list = []
        # iterate over the bands
        for band in bands:
            # iterate over the files in file_names
            for file in file_names:
                # get the date of the file and the band of the file using the grouping function
                file_date, file_band = grouping_function(file)
                # compare the current date and band from iteration with the date and band from the grouping_function,
                # if the dates and bands match, we have the right file
                if date == file_date and band == file_band:
                    # read it using read_satellite_file and append it to the band dimension list
                    band_list.append(read_satellite_file(file))
                    break
                    # break
                    
        # np.stack the band dimension list and append it to the date dimension list
        date_list.append(np.stack(band_list))
    # np.stack the date dimension list to create a 4D array with dimensions (date, band, height, width)
    new_data = np.stack(date_list)

    # An xarray is a wrapper for the np.ndarray, allowing us to store data about data
    # (essentially metadata). An xarray has dimensions and coordinates.
    #
    # dimensions (dims):
    #   - dimensions represent the axes or directions along which data is organized.
    #   - each dimension has a name, serving as an identifier for that axis.
    #   - they can be 0-dimensional (scalar), 1-dimensional (e.g., time), 2-dimensional (e.g., latitude and longitude), or higher-dimensional.
    #   - dimensions define the shape of the array, allowing for intuitive referencing and slicing.
    #
    # coordinates (coords):
    #   - coordinates are values associated with specific points along the dimensions of the array.
    #   - they provide metadata labeling individual points of the data array.
    #   - coordinates can be 1D arrays, scalars, or multidimensional arrays aligned with the dataset's dimensions.
    #   - examples include timestamps for time-series data, geographical coordinates for spatial data, or other relevant auxiliary variables.
    #   - they enable easy selection, slicing, and alignment of data based on meaningful labels rather than numerical indices.
    #   - xarray also supports auxiliary coordinate variables, providing additional information about data points without serving as dimensions.
    #
    # So for our data. we can access it like numpy using data_array[time][band][height][width], or we can also access it in more
    # advanced methods: https://docs.xarray.dev/en/latest/user-guide/indexing.html
    #
    # To provide an example of advanced accessing using our data, using the Sentinel 1 data_array, we can do
    # data_array[data_array.sel(date="20200723", band="VH")
    #
    #
    #
    # create the xr.DataArray with dimensions (dim):
    #   ("date", "band", "height", "width")
    # and the corresponding coordinates (coords):
    #   {"date": dates, "band": bands, "height": image height, "width": image width}
    #
    # For dims:
    #   We can just pass the string of tuples like above to label our dimensions
    #
    # For coordinates:
    #   dates and bands will be the lists from above
    #
    #   image height and image width can be grabbed from the last two indicies of the 4D np.ndarray .shape
    #   and put into a range()
    #   (Note: while the range is technically not needed, we will be doing it for consistency)
    

    # add the satellite_type, tile_dir, and parent_tile_id as attributes to the xarray
    dataArray = xr.DataArray(
        new_data,
        dims=("date", "band", "height", "width"),
        coords={
            "date": dates,
            "band": bands,
            "height": range(new_data.shape[-2]),
            "width": range(new_data.shape[-1]),
        },
    )
    dataArray = dataArray.assign_attrs(satellite_type=satellite_type.value, tile_dir=str(tile_dir), parent_tile_id=parent_tile_id)
    return dataArray


def load_satellite_list(
    tile_dir: Path, satellite_type_list: List[SatelliteType]
) -> "list[xr.DataArray]":
    """
    Loads all the satellites from the tile directory based on the satellite type list.

    Parameters
    ----------
    tile_dir : Path
        The Tile directory containing the satellite tif files.
    satellite_type_list : List[SatelliteType]
        List of the type of satellite, each should be one of SatelliteType.VIIRS,
        SatelliteType.S1, SatelliteType.S2, SatelliteType.LANDSAT, or SatelliteType.GT.

    Returns
    -------
    List[xr.DataArray]
        List of data arrays for each SatelliteType in the satellite type list.
    """
    satellite_list = []
    for satellite_type in satellite_type_list:
        satellite_list.append(load_satellite(tile_dir, satellite_type))
    return satellite_list


# given
def load_satellite_dir(
    data_dir: Path, satellite_type_list: List[SatelliteType]
) -> List[List[xr.DataArray]]:
    """
    Load all bands for a given satellite type from a directory of multiple
    tile files.

    Parameters
    ----------
    data_dir : Path
        The directory containing all of the satellite tiles.
    satellite_type_list : List[SatelliteType]
        A list of the type of satellite, should be one of SatelliteType.VIIRS,
        SatelliteType.S1, SatelliteType.S2, SatelliteType.LANDSAT, or SatelliteType.GT.
    Returns
    -------
    List[List[xr.DataArray]]
        A list of tiles, each element containing a satellite_list (list of each satellites data_array
        for that tile).
    """
    return [
        load_satellite_list(tile_dir, satellite_type_list)
        for tile_dir in sorted(data_dir.iterdir())
        if tile_dir.is_dir()
    ]


def create_satellite_dataset_list(
    list_of_data_array_list: List[List[xr.DataArray]],
    satellite_type_list: List[SatelliteType],
    list_of_preprocess_func_list: List[List[Callable]] = None,
):
    """
    Creates the satellite_dataset_list of all the data (gotten from load_satellite_dir). This function takes
    the data arrays from each tile and combines them into a single dataset by satellite.
    Each satellite gets its own xr.Dataset: https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html

    Parameters
    ----------
    list_of_data_array_list : List[List[xr.DataArray]]
        A list of tiles, each element containing a satellite_list (list of each satellites data_array
        for that tile)
    satellite_type_list : List[SatelliteType]
        A list of the type of satellite, should be one of SatelliteType.VIIRS,
        SatelliteType.S1, SatelliteType.S2, SatelliteType.LANDSAT, or SatelliteType.GT.
    list_of_preprocess_func_list :  List[List[Callable]]
        A list of lists. Inside every list, each element is a function to apply to the data_array
        as a preprocessing function.

    Returns
    -------
    List[xr.Dataset]
        List of xr.Dataset, where each element is the data for a single satellite over all the tiles.
    """
    # create dicts
    data_dict_list = [dict() for _ in satellite_type_list]
    # iterate over each tile
    for satellite_list in list_of_data_array_list:
        # iterate through an enumeration over each satellite_list
        for index, data_array in enumerate(satellite_list):
            # if the list_of_preprocessing_func_list exists
            if list_of_preprocess_func_list != None:
                # if the sublist at this index exists
                if list_of_preprocess_func_list[index] != None:
                    # apply each function in the sublist to the data_array
                    for func in list_of_preprocess_func_list[index]:
                        data_array = func(data_array)
            # add the data array to the proper data_dict using the parent_tile_id as the key
            # and set the value to be the data_array
            (data_dict_list[index])[data_array.attrs["parent_tile_id"]] = data_array

    # This can be kind of confusing to understand, so what are we exactly doing here and why?
    # First, we are creating a list of data_dicts for each satellite type. We need to do this initially
    # for optimal iteration (We don't want to loop through the entire dataset more than once).
    # Second, we iterate over every tile. For each tile, we add the data_array of that tile to its data_dict,
    # so the VIIRS data_dict will contain only the data_array(s) with the satellite_type attribute for
    # VIIRS (and the same for every other satellite). This will allow us to create a dataset for each satellite.
    # The dataset will contain all the tiles for that satellite and that satellite only. You can think of it like
    # it the shape (tile, date, band, height, width).

    # --- start here ---
    # create a list to store the data sets
    datasets_list = []
    # iterate through an enumeration over the data_dict_list
    for index, data_dict in enumerate(data_dict_list):
    # you should have an index and a data_dict in the for loop
        dataset = xr.Dataset(data_dict)
        # create the xr.Dataset and pass the data_dict
        
        # set the dataset attribute "satellite_type" to be the value of the satellite_type_list[index]
        dataset.attrs["satellite_type"] = satellite_type_list[index].value
        # append the dataset to the data list
        datasets_list.append(dataset)
    # return the data_set_list
    return datasets_list
