import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import xarray as xr

# local modules
sys.path.append(".")
from src.utilities import SatelliteType


class Subtile:
    def __init__(
        self,
        satellite_list: List[xr.DataArray],
        ground_truth: xr.DataArray | None = None,
        slice_size: tuple = (4, 4),
        parent_tile_id: int = None
    ):
        """
        This class handles saving and loading of subtiles for a parent image (Tile#).

        Parameters:
            slice_size: the slice size of the image ( (4, 4) = 4x4 subtiles )
        """
        self.satellite_list = satellite_list
        self.ground_truth = ground_truth
        self.slice_size = slice_size
        self.parent_tile_id = satellite_list[0].attrs["parent_tile_id"] if parent_tile_id == None else parent_tile_id

    def __calculate_slice_index(self, x: int, y: int, slice_size: tuple, length: tuple):
        # calculate the start and end indices for the slice based
        # on the slice_size and the x and y coordinates
        start_index = (
            int(np.divide(length[0], slice_size[0]) * x),
            int(np.divide(length[1], slice_size[1]) * y),
        )
        end_index = (
            int(np.divide(length[0], slice_size[0]) * (x + 1)),
            int(np.divide(length[1], slice_size[1]) * (y + 1)),
        )

        if start_index[0] > length[0] or start_index[1] > length[1]:
            raise IndexError(
                f"Start index {start_index} out of range for img of shape {length}"
            )

        if end_index[0] > length[0] or end_index[1] > length[1]:
            raise IndexError(
                f"End index {end_index} out of range for img of shape {length}"
            )

        return start_index, end_index

    def get_subtile_from_parent_image(self, x: int, y: int) -> xr.Dataset:
        """
        Subtiles the data by the given slice_size at x, y

        Parameters:
            x: x index of the subtile
            y: y index of the subtile
            parent_image_dataset: the xr.Dataset {img, label} of the parent image
        Returns:
           xr.Dataset {img, label} of the subtile
        """
        img_length = (self.satellite_list[0])[0][0].shape
        label_length = self.ground_truth[0][0].shape

        start_index_img, end_index_img = self.__calculate_slice_index(
            x, y, self.slice_size, img_length
        )
        start_index_label, end_index_label = self.__calculate_slice_index(
            x, y, self.slice_size, label_length
        )

        new_satellite_list = list()
        for data_array in self.satellite_list:
            sliced_data_array = data_array[
                :,
                :,
                start_index_img[0] : end_index_img[0],
                start_index_img[1] : end_index_img[1],
            ]
            sliced_data_array.attrs["x"] = x
            sliced_data_array.attrs["y"] = y

            new_satellite_list.append(sliced_data_array)

        new_ground_truth = self.ground_truth[
            :,
            :,
            start_index_label[0] : end_index_label[0],
            start_index_label[1] : end_index_label[1],
        ]
        new_ground_truth.attrs["x"] = x
        new_ground_truth.attrs["y"] = y

        return (
            new_satellite_list,
            new_ground_truth,
        )

    def _save_image(
        self, subtiled_data_array: xr.DataArray, subtile_directory: Path, x: int, y: int
    ):
        subtiled_data_array.to_netcdf(
            subtile_directory
            / subtiled_data_array.attrs["parent_tile_id"]
            / f"{x}_{y}"
            / f"{subtiled_data_array.attrs['satellite_type']}.nc"
        )

    def _save_label(
        self,
        subtiled_ground_truth: xr.DataArray,
        subtile_directory: Path,
        x: int,
        y: int,
    ):
        subtiled_ground_truth.to_netcdf(
            subtile_directory
            / subtiled_ground_truth.attrs["parent_tile_id"]
            / f"{x}_{y}"
            / f"{SatelliteType.GT.value}.nc"
        )

    def save(self, directory_to_save: Path) -> None:
        """
        Saves the subtiles of the parent tile to:
            img goes to directory_to_save / subtiles / images
            label goes to directory_to_save/ subtiles / labels
            (the image and label are assumed to have the same resolution)
        Saved files follow the naming scheme {parent_tile_id}_{x}_{y}.npy

        Parameters:
            processed_dict: the {img, label} dict to save the subtiles with
            directory_to_save: the directory where the images and labels will be saved
                               as described above
        Returns:
            None
        """
        directory_to_save.mkdir(parents=True, exist_ok=True)

        subtile_directory = directory_to_save / "subtiles"
        subtile_directory.mkdir(parents=True, exist_ok=True)

        # iterate over the slice_size
        for x in range(self.slice_size[0]):
            for y in range(self.slice_size[1]):
                subtiled_list, subtiled_ground_truth = (
                    self.get_subtile_from_parent_image(x, y)
                )

                Path(subtile_directory / self.parent_tile_id).mkdir(exist_ok=True)
                assert Path(subtile_directory / self.parent_tile_id).exists()

                Path(subtile_directory / self.parent_tile_id / f"{x}_{y}").mkdir(
                    exist_ok=True
                )
                assert Path(
                    subtile_directory / self.parent_tile_id / f"{x}_{y}"
                ).exists()

                # save the subtile of the image at the images directory
                for subtiled_data_array in subtiled_list:
                    # 4D (time, band, height, width)
                    self._save_image(subtiled_data_array, subtile_directory, x, y)
                # save the subtile of the label at the labels directory
                self._save_label(subtiled_ground_truth, subtile_directory, x, y)
        # clear the data because it has been saved into the subtiled files
        self.satellite_list = None
        self.ground_truth = None

    def load_subtile(
        self,
        directory_to_load: Path,
        satellite_type_list: List[SatelliteType],
        x: int,
        y: int,
    ) -> List[xr.DataArray]:
        """
        Loads a subtile file ({parent_tile_id}_{x}_{y}.npy)

        Parameters:
            subtile_file: path to the subtile file
        Returns:
            List[xr.DataArray]
        """
        tile_dir = directory_to_load / "subtiles" / self.parent_tile_id
        list_of_subtiled_data_array = list()
        for satellite_type in satellite_type_list:
            subtile_file = tile_dir / f"{x}_{y}" / f"{satellite_type.value}.nc"
            assert subtile_file.exists() == True, f"{subtile_file} does not exist"

            data_array = xr.load_dataarray(subtile_file)
            assert data_array.attrs["x"] == np.int32(x), f"{data_array.attrs['x']}, {x}"
            assert data_array.attrs["y"] == np.int32(y), f"{data_array.attrs['y']}, {y}"

            list_of_subtiled_data_array.append(data_array)
        return list_of_subtiled_data_array

    @staticmethod
    def load_subtile_by_dir(
        directory_to_load: Path,
        satellite_type_list: List[SatelliteType],
        slice_size: Tuple[int, int] = (4, 4),
        has_gt: bool = True,
    ) -> List[xr.DataArray]:
        """
        Loads a directory of subtile files ({parent_tile_id}_{x}_{y}.npy)

        Parameters:
            directory_to_load: Path to the subtile file directory
            satellite_type_list: list of satellites to load
            slice_size: slice size of the subtile
        Returns:
            List[xr.DataArray]
        """
        list_of_subtiled_data_array = list()
        for satellite_type in satellite_type_list:
            subtile_file = directory_to_load / f"{satellite_type.value}.nc"
            assert subtile_file.exists() == True, f"{subtile_file} does not exist"

            data_array = xr.load_dataarray(subtile_file)

            list_of_subtiled_data_array.append(data_array)

        if has_gt:
            gt_data_array = xr.load_dataarray(
                directory_to_load / f"{SatelliteType.GT.value}.nc"
            )
        else:
            gt_data_array = None

        subtile = Subtile(
            satellite_list=list_of_subtiled_data_array,
            ground_truth=gt_data_array,
            slice_size=slice_size,
        )
        return subtile

    def restitch(
        self, directory_to_load: Path, satellite_type_list: List[SatelliteType]
    ) -> None:
        """
        Restitiches the subtile images to their original image

        Parameters:
            directory_to_load: path to the directory where the subtile images and labels are loaded from
            satellite_type_list: list of satellite types that dictates which satellites will be loaded

        Returns:
            result: Tuple containing:
                restitched_image: List[xr.DataArray]
                restitched_label: xr.DataArray
        """
        # even though this is slightly less efficient that iterating over once,
        # it's way more readable and understandable for everyone

        # add the ground truth to the satellite_type_list so this
        # operation below will retrieve it for us
        satellite_type_list_with_gt = satellite_type_list + [SatelliteType.GT]

        list_of_data_array = list()
        for satellite_type in satellite_type_list_with_gt:
            row = []
            for x in range(self.slice_size[0]):
                col = []
                for y in range(self.slice_size[1]):
                    data_array = self.load_subtile(
                        directory_to_load, [satellite_type], x, y
                    )[0]
                    # remove the subtile attributes now that they are no longer needed
                    del data_array.attrs["x"]
                    del data_array.attrs["y"]

                    col.append(data_array)
                row.append(xr.concat(col, dim="width"))
            data_array = xr.concat(row, dim="height")
            list_of_data_array.append(data_array)

        self.satellite_list = list_of_data_array[:-1]
        self.ground_truth = list_of_data_array[-1]
