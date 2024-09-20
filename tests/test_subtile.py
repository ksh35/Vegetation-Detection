import sys
import unittest
from pathlib import Path
from shutil import rmtree

import numpy as np
import xarray as xr

# local modules
sys.path.append(".")
from src.preprocessing.file_utils import (
    load_satellite,
    load_satellite_dir,
    load_satellite_list,
)
from src.preprocessing.subtile import *
from src.utilities import SatelliteType

ROOT = Path.cwd()


def check_data_array_is_valid(
    self: unittest.TestCase,
    data_array: xr.DataArray,
    satellite_type: SatelliteType,
    dates: List[str],
    bands: List[str],
    shape: tuple,
):
    """
    This function is used as a helper for the subtile tests
    """
    # check object and shape
    self.assertIsInstance(data_array, xr.DataArray)
    self.assertEqual(data_array.shape, shape)

    # check dimensions
    self.assertEqual(len(data_array.dims), 4)
    self.assertEqual(data_array.dims, ("date", "band", "height", "width"))

    # check attributes
    self.assertEqual(len(data_array.attrs.keys()), 3, f"{data_array.attrs.keys()}")
    for attr in ["satellite_type", "tile_dir", "parent_tile_id"]:
        self.assertIn(attr, data_array.attrs)

    # check satellite type
    self.assertIsInstance(data_array.attrs["satellite_type"], str)
    self.assertEqual(data_array.attrs["satellite_type"], satellite_type.value)

    # check tile_dir
    self.assertIsInstance(data_array.attrs["tile_dir"], str)
    self.assertEqual(data_array.attrs["tile_dir"], str(self.tile_dir))

    # check parent_tile_id
    self.assertIsInstance(data_array.attrs["parent_tile_id"], str)
    self.assertEqual(data_array.attrs["parent_tile_id"], "Tile1")

    # check access by date
    self.assertEqual(len(data_array["date"]), shape[0])
    # check access by band
    self.assertEqual(len(data_array["band"]), shape[1])
    # ensure dates and bands line up correctly
    for index, date in enumerate(dates):
        self.assertEqual(date, data_array["date"].values[index])
    for index, band in enumerate(bands):
        self.assertEqual(band, data_array["band"].values[index])


class TestSubtile(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.data_dir = ROOT / "data" / "raw" / "Train"
        self.tile_dir = self.data_dir / "Tile1"
        self.save_dir = ROOT / "data" / "processed"

        # you can change this at will, just cannot make it greater than 16
        # since our GT is 16x16
        self.slice_size = (4, 4)
        assert self.slice_size[0] <= 16
        assert self.slice_size[1] <= 16

        self.satellite_type_list = [
            SatelliteType.VIIRS,
            SatelliteType.S1,
            SatelliteType.S2,
            SatelliteType.LANDSAT,
        ]
        self.subtile_shapes = [
            (
                9,
                1,
                np.divide(800, self.slice_size[0]),
                np.divide(800, self.slice_size[1]),
            ),
            (
                4,
                2,
                np.divide(800, self.slice_size[0]),
                np.divide(800, self.slice_size[1]),
            ),
            (
                4,
                12,
                np.divide(800, self.slice_size[0]),
                np.divide(800, self.slice_size[1]),
            ),
            (
                3,
                11,
                np.divide(800, self.slice_size[0]),
                np.divide(800, self.slice_size[1]),
            ),
        ]
        self.restiched_shapes = [
            (9, 1, 800, 800),
            (4, 2, 800, 800),
            (4, 12, 800, 800),
            (3, 11, 800, 800),
        ]

        assert len(self.satellite_type_list) == len(
            self.subtile_shapes
        ), f"self.satellite_type_list and self.subtile_shapes must have the same length. Currently {len(self.satellite_type_list)} and {len(self.subtile_shapes)}"

    def tearDown(self) -> None:
        # remove the data after every test
        if self.save_dir.exists():
            rmtree(self.save_dir)

    @classmethod
    def tearDownClass(self) -> None:
        # remove the data after every test
        if self.save_dir.exists():
            rmtree(self.save_dir)

    def test_initialization(self):
        satellite_list = load_satellite_list(self.tile_dir, self.satellite_type_list)
        subtile = Subtile(
            satellite_list,
            ground_truth=load_satellite(self.tile_dir, SatelliteType.GT),
            slice_size=(1, 1),
        )
        self.assertIsInstance(subtile.satellite_list, list)
        self.assertEqual(len(subtile.satellite_list), len(self.satellite_type_list))

        self.assertIsInstance(subtile.ground_truth, xr.DataArray)

        self.assertEqual(subtile.slice_size, (1, 1))

        self.assertEqual(subtile.parent_tile_id, "Tile1")

    def test_save_single_tile(self):
        satellite_list = load_satellite_list(self.tile_dir, self.satellite_type_list)
        subtile = Subtile(
            satellite_list,
            ground_truth=load_satellite(self.tile_dir, SatelliteType.GT),
            slice_size=self.slice_size,
        )
        subtile.save(self.save_dir)

        subtiled_directory = self.save_dir / "subtiles" / "Tile1"
        self.assertTrue(subtiled_directory.exists())
        # make sure the expected number of files exists
        # (each satellite + gt) * slice_size[0] * slice_size[1]
        self.assertEqual(
            len([*Path(subtiled_directory).glob("*/*.nc")]),
            (len(self.satellite_type_list) + 1)
            * self.slice_size[0]
            * self.slice_size[1],
        )

        # make sure the data is no longer stored in the subtiles
        self.assertIsNone(subtile.satellite_list)
        self.assertIsNone(subtile.ground_truth)

    def test_load_single_tile(self):
        satellite_list = load_satellite_list(self.tile_dir, self.satellite_type_list)
        subtile = Subtile(
            satellite_list,
            ground_truth=load_satellite(self.tile_dir, SatelliteType.GT),
            slice_size=self.slice_size,
        )
        subtile.save(self.save_dir)

        # check output
        for x in range(self.slice_size[0]):
            for y in range(self.slice_size[1]):
                # load each subtile
                subtile_list = subtile.load_subtile(
                    self.save_dir,
                    self.satellite_type_list,
                    x,
                    y,
                )

                # check each satellite
                for index, data_array in enumerate(subtile_list):
                    self.assertEqual(
                        data_array.shape,
                        self.subtile_shapes[index],
                    )

                # check ground truth
                gt_data_array = subtile.load_subtile(
                    self.save_dir, [SatelliteType.GT], x, y
                )[0]
                self.assertEqual(
                    gt_data_array.shape,
                    (
                        1,
                        1,
                        np.divide(16, self.slice_size[0]),
                        np.divide(16, self.slice_size[1]),
                    ),
                )

    def test_save_dir(self):
        if False:
            for x in range(1, 61):
                tile_dir = self.data_dir / f"Tile{x}"
                satellite_list = load_satellite_list(tile_dir, self.satellite_type_list)
                subtile = Subtile(
                    satellite_list,
                    ground_truth=load_satellite(tile_dir, SatelliteType.GT),
                    slice_size=self.slice_size,
                )
                subtile.save(self.save_dir)

                subtiled_directory = self.save_dir / "subtiles" / f"Tile{x}"
                self.assertTrue(subtiled_directory.exists())
                # make sure the expected number of files exists
                # (each satellite + gt) * slice_size[0] * slice_size[1]
                self.assertEqual(
                    len([*Path(subtiled_directory).glob("*.nc")]),
                    (len(self.satellite_type_list) + 1)
                    * self.slice_size[0]
                    * self.slice_size[1],
                )

                # make sure the data is no longer stored in the subtiles
                self.assertIsNone(subtile.satellite_list)
                self.assertIsNone(subtile.ground_truth)

    def test_load_dir(self):
        if False:
            for t in range(1, 61):
                tile_dir = self.data_dir / f"Tile{t}"
                satellite_list = load_satellite_list(tile_dir, self.satellite_type_list)
                subtile = Subtile(
                    satellite_list,
                    ground_truth=load_satellite(tile_dir, SatelliteType.GT),
                    slice_size=self.slice_size,
                )
                subtile.save(self.save_dir)

                # check output
                for x in range(self.slice_size[0]):
                    for y in range(self.slice_size[1]):
                        # load each subtile
                        subtile_list = subtile.load_subtile(
                            self.save_dir,
                            self.satellite_type_list,
                            x,
                            y,
                        )

                        # check each satellite
                        for index, data_array in enumerate(subtile_list):
                            self.assertEqual(
                                data_array.shape,
                                self.subtile_shapes[index],
                            )

                        # check ground truth
                        gt_data_array = subtile.load_subtile(
                            self.save_dir, [SatelliteType.GT], x, y
                        )[0]
                        self.assertEqual(
                            gt_data_array.shape,
                            (
                                1,
                                1,
                                np.divide(16, self.slice_size[0]),
                                np.divide(16, self.slice_size[1]),
                            ),
                        )

    def test_restitch(self):
        satellite_list = load_satellite_list(self.tile_dir, self.satellite_type_list)
        subtile = Subtile(
            satellite_list,
            ground_truth=load_satellite(self.tile_dir, SatelliteType.GT),
            slice_size=self.slice_size,
        )
        subtile.save(self.save_dir)

        # make sure the data is no longer stored in the subtiles
        self.assertIsNone(subtile.satellite_list)
        self.assertIsNone(subtile.ground_truth)

        # restitch tile
        subtile.restitch(self.save_dir, self.satellite_type_list)

        self.assertIsInstance(subtile.satellite_list, list)
        self.assertIsInstance(subtile.ground_truth, xr.DataArray)
        self.assertEqual(len(subtile.satellite_list), len(self.satellite_type_list))

        # check each satellite
        for index, data_array in enumerate(subtile.satellite_list):
            self.assertIsInstance(data_array, xr.DataArray)
            self.assertEqual(data_array.shape, self.restiched_shapes[index])

        # check ground truth
        self.assertEqual(subtile.ground_truth.shape, (1, 1, 16, 16))

    def test_restitch_with_data(self):
        satellite_type = SatelliteType.S1

        # Create a 2D Image for the dates and bands to store
        resolution = (400, 400)
        image_2d = np.ones(resolution)
        length, slice_size = image_2d.shape, (2, 2)

        value = 0
        for x in range(slice_size[0]):
            for y in range(slice_size[1]):
                start_index = (
                    int(np.divide(length[0], slice_size[0]) * x),
                    int(np.divide(length[1], slice_size[1]) * y),
                )
                end_index = (
                    int(np.divide(length[0], slice_size[0]) * (x + 1)),
                    int(np.divide(length[1], slice_size[1]) * (y + 1)),
                )
                image_2d[
                    start_index[0] : end_index[0], start_index[1] : end_index[1]
                ] = value
                value += 1

        # image_2d is now an image like below:
        #    0  1  2  3
        #    4  5  6  7
        #    8  9  10 11
        #    12 13 14 15
        #
        # where each integer is a (resolution / slice size) chunk of that integer.
        # So if the resolution is (400, 400) and the slice size is (4, 4), each chunk
        # of integer Z would be a (100, 100) array of Z's

        # creates identical mock dates and bands dimensions and values for the image.
        # We are stacking the image over and over just to make it look like our data.
        date_dim = list()
        for _ in range(3):
            band_dim = list()
            for _ in range(2):
                band_dim.append(image_2d)
            date_dim.append(np.stack(band_dim))
        image_4d = np.stack(date_dim)

        # Create a custom data array to test the restitching
        data_array = xr.DataArray(
            data=image_4d,
            dims=("date", "band", "height", "width"),
            coords={
                "date": ["1", "2", "3"],
                "band": ["1", "2"],
                "height": range(image_4d.shape[2]),
                "width": range(image_4d.shape[3]),
            },
            attrs={
                "satellite_type": satellite_type.value,
                "tile_dir": str(self.tile_dir),
                "parent_tile_id": "Tile1",
            },
        )

        # Create a custom ground truth to test the restitching
        gt_array = xr.DataArray(
            data=np.reshape(image_2d, (1, 1, 400, 400)),
            dims=("date", "band", "height", "width"),
            coords={
                "date": ["1"],
                "band": ["1"],
                "height": range(image_4d.shape[2]),
                "width": range(image_4d.shape[3]),
            },
            attrs={
                "satellite_type": SatelliteType.GT.value,
                "tile_dir": str(self.tile_dir),
                "parent_tile_id": "Tile1",
            },
        )

        # save the custom data_array and ground truth
        subtile = Subtile([data_array], gt_array, slice_size=slice_size)
        subtile.save(self.save_dir)

        # Assert that data saved correctly, and directory length matches expected length.
        # Expected length is (# satellites + gt) * slice size 0 * slice size 1
        subtiled_directory = Path(self.save_dir / "subtiles" / "Tile1")
        self.assertTrue(subtiled_directory.exists())
        self.assertEqual(
            len([*Path(subtiled_directory).glob("*/*.nc")]),
            (1 + 1) * slice_size[0] * slice_size[1],
        )

        # make sure the data is no longer stored in the subtiles
        self.assertIsNone(subtile.satellite_list)
        self.assertIsNone(subtile.ground_truth)

        # restitch tile
        subtile.restitch(self.save_dir, [satellite_type])

        # validate return types
        self.assertIsInstance(subtile.satellite_list, list)
        self.assertEqual(len(subtile.satellite_list), 1)
        self.assertIsInstance(subtile.ground_truth, xr.DataArray)

        for data_array in subtile.satellite_list:
            # validate types
            check_data_array_is_valid(
                self,
                data_array,
                satellite_type,
                ["1", "2", "3"],
                ["1", "2"],
                (3, 2, length[0], length[1]),
            )
            # check data is the same as before it was subtiled
            self.assertTrue(np.array_equiv(data_array.values, image_2d))

        # validate types for gt
        check_data_array_is_valid(
            self,
            subtile.ground_truth,
            SatelliteType.GT,
            ["1"],
            ["1"],
            (1, 1, length[0], length[1]),
        )
        # check data is the same as before it was subtiled
        self.assertTrue(np.array_equiv(subtile.ground_truth.values, image_2d))


if __name__ == "__main__":
    unittest.main()
