import sys
import unittest
from pathlib import Path

from torchvision import transforms
from shutil import rmtree, copytree
import xarray as xr

sys.path.append(".")
from src.esd_data.augmentations import (
    AddNoise,
    Blur,
    RandomHFlip,
    RandomVFlip,
    ToTensor,
)
from src.esd_data.datamodule import ESDDataModule
from src.esd_data.dataset import ESDDataset
from src.utilities import SatelliteType


def check_data_array_is_valid(
    self: unittest.TestCase,
    data_array: xr.DataArray,
    satellite_type: SatelliteType,
):
    """
    This function is used as a helper for the test_load_satellite tests
    """
    # check object and shape
    self.assertIsInstance(data_array, xr.DataArray)
    # self.assertEqual(data_array.shape, shape)

    # check dimensions
    self.assertEqual(len(data_array.dims), 4)
    self.assertEqual(data_array.dims, ("date", "band", "height", "width"))

    # check attributes
    self.assertEqual(len(data_array.attrs.keys()), 3)
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


ROOT = Path.cwd()


class TestESDDataModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processed_dir = ROOT / "data" / "processed" / "unit_test"
        cls.raw_dir = ROOT / "data" / "raw" / "unit_test"

        if cls.processed_dir.exists():
            rmtree(cls.processed_dir)
        if cls.raw_dir.exists():
            rmtree(cls.raw_dir)

        (cls.raw_dir).mkdir(exist_ok=True)
        test_tiles = [item for item in sorted((ROOT / "data" / "raw" / "Train").iterdir()) if item.is_dir()][:10]
        assert len(test_tiles) == 10
        for tile in test_tiles:
            copytree(tile, cls.raw_dir / tile.name)

        cls.tile_dir = cls.raw_dir / "Tile1"
        cls.batch_size = 32
        cls.seed = 12378921
        cls.selected_bands = {
            SatelliteType.VIIRS: ["0"],
            SatelliteType.S1: ["VV", "VH"],
            SatelliteType.S2: ["04", "03", "02"],
            SatelliteType.LANDSAT: ["5", "4", "3"],
        }
        cls.slice_size = (4, 4)
        cls.train_size = 0.8
        cls.transform = transforms.Compose(
            [
                transforms.RandomApply([AddNoise()], p=0.5),
                transforms.RandomApply([Blur()], p=0.5),
                transforms.RandomApply([RandomHFlip()], p=0.5),
                transforms.RandomApply([RandomVFlip()], p=0.5),
                ToTensor(),
            ]
        )

        cls.satellite_type_list = [
            SatelliteType.VIIRS,
            SatelliteType.S1,
            SatelliteType.S2,
            SatelliteType.LANDSAT,
        ]

    def setUp(self):
        self.data_module = ESDDataModule(
            processed_dir=self.processed_dir,
            raw_dir=self.raw_dir,
            batch_size=self.batch_size,
            seed=self.seed,
            selected_bands=self.selected_bands,
            train_size=self.train_size,
            transform_list=self.transform,
        )

    @classmethod
    def tearDownClass(cls):
        if cls.processed_dir.exists():
            rmtree(cls.processed_dir)
        if cls.raw_dir.exists():
            rmtree(cls.raw_dir)

    def test_initialization(self):
        self.assertIsInstance(self.data_module.processed_dir, Path)
        self.assertEqual(self.data_module.processed_dir, self.processed_dir)
        self.assertIsInstance(self.data_module.raw_dir, Path)
        self.assertEqual(self.data_module.raw_dir, self.raw_dir)
        self.assertIsInstance(self.data_module.batch_size, int)
        self.assertEqual(self.data_module.batch_size, self.batch_size)
        self.assertIsInstance(self.data_module.seed, int)
        self.assertEqual(self.data_module.seed, self.seed)
        self.assertIsInstance(self.data_module.transform, transforms.transforms.Compose)
        self.assertEqual(self.data_module.selected_bands, self.selected_bands)

    def test_load_and_preprocess(self):
        data_array_list, gt_data_array = self.data_module.load_and_preprocess(
            self.raw_dir / "Tile1",
        )

        self.assertIsInstance(data_array_list, list)
        for index, data_array in enumerate(data_array_list):
            check_data_array_is_valid(self, data_array, self.satellite_type_list[index])

        check_data_array_is_valid(self, gt_data_array, SatelliteType.GT)

    def test_prepare_data(self):
        self.data_module.prepare_data()
        for directory in ["Train", "Val"]:
            self.assertNotEqual(
                len([*(self.processed_dir / directory / "subtiles").iterdir()]), 0
            )
            for tile in (self.processed_dir / directory / "subtiles").iterdir():
                self.assertEqual(
                    len([*tile.glob("*/*.nc")]),
                    (
                        self.slice_size[0]
                        * self.slice_size[1]
                        * (len(self.satellite_type_list) + 1)
                    ),
                )

    def test_setup(self):
        self.data_module.setup("fit")

        self.assertIsInstance(
            self.data_module.train_dataset,
            ESDDataset,
        )
        self.assertIsInstance(
            self.data_module.train_dataset,
            ESDDataset,
        )

    def test_train_dataloader(self):
        self.data_module.setup("fit")
        self.assertTrue(callable(self.data_module.train_dataloader))

    def test_val_dataloader(self):
        self.data_module.prepare_data()
        self.data_module.setup("fit")
        self.assertTrue(callable(self.data_module.val_dataloader))


if __name__ == "__main__":
    unittest.main()
