"""FIXME: legacy tests"""

import sys

import unittest
import torch
import numpy as np
from pathlib import Path
from torchvision import transforms
from shutil import rmtree

sys.path.append(".")
from src.esd_data.dataset import (
    ESDDataset,
)
from src.preprocessing.subtile import Subtile
from src.utilities import SatelliteType
from src.preprocessing.file_utils import load_satellite, load_satellite_list
from src.esd_data.augmentations import (
    AddNoise,
    Blur,
    RandomHFlip,
    RandomVFlip,
    ToTensor,
)


ROOT = Path.cwd()


class TestESDDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up any necessary objects, paths, or configurations for testing
        cls.processed_dir = ROOT / "data" / "processed" / "unit_test"
        cls.raw_dir = ROOT / "data" / "raw" / "Train"

        if cls.processed_dir.exists():
            rmtree(cls.processed_dir)

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

        cls.slice_size = (4, 4)

        cls.subtiles = [
            Subtile(
                satellite_list=load_satellite_list(
                    cls.raw_dir / "Tile1", cls.satellite_type_list
                ),
                ground_truth=load_satellite(cls.raw_dir / "Tile1", SatelliteType.GT),
                slice_size=cls.slice_size,
            ),
        ]

        for subtile in cls.subtiles:
            subtile.save(cls.processed_dir / "Train")

    def setUp(self) -> None:
        self.dataset = ESDDataset(
            processed_dir=self.processed_dir / "Train",
            transform=self.transform,
            satellite_type_list=self.satellite_type_list,
            slice_size=self.slice_size,
        )

    @classmethod
    def tearDownClass(cls):
        if cls.processed_dir.exists():
            rmtree(cls.processed_dir)

    def test_initialization(self):
        dataset = self.dataset
        self.assertIsInstance(dataset.subtile_dirs, list)
        for subtile_dir in dataset.subtile_dirs:
            self.assertIsInstance(subtile_dir, Path)
            self.assertEqual(
                subtile_dir.parent, self.processed_dir / "Train" / "subtiles" / "Tile1"
            )

        self.assertIsInstance(dataset.transform, transforms.transforms.Compose)

        self.assertIsInstance(dataset.satellite_type_list, list)
        for sat_type in dataset.satellite_type_list:
            self.assertIsInstance(sat_type, SatelliteType)
        self.assertEqual(
            len(dataset.satellite_type_list), len(self.satellite_type_list)
        )

        self.assertIsInstance(dataset.slice_size, tuple)
        for slice in dataset.slice_size:
            self.assertIsInstance(slice, int)
        self.assertEqual(dataset.slice_size, self.slice_size)

    def test_len_method(self):
        dataset = self.dataset
        self.assertEqual(len(dataset), 16)

    def test_aggregate_time_method(self):
        dataset = self.dataset
        # Test the __aggregate_time method
        input_data = np.random.rand(5, 2, 10, 10)  # Example input data
        output_data = dataset._ESDDataset__aggregate_time(input_data)
        expected_shape = (5 * 2, 10, 10)
        self.assertEqual(
            output_data.shape,
            expected_shape,
            "Aggregate time does not have the correct shape.",
        )

    def test_getitem_index_0(self):
        dataset = self.dataset
        idx = 0
        X, y = dataset.__getitem__(idx)
        self.assertIsInstance(
            X,
            torch.Tensor,
            "The __getitem__ method is not returning the expected type for X",
        )

        self.assertEqual(
            X.shape,
            (
                98,
                np.divide(800, self.slice_size[0]),
                np.divide(800, self.slice_size[1]),
            ),
            f"X's shape is not the right size. Current: f{X.shape}, Expected: (98, {np.divide(800, self.slice_size[0])}, {np.divide(800, self.slice_size[1])})",
        )

        self.assertIsInstance(
            y,
            torch.Tensor,
            "The __getitem__ method is not returning the expected type for y",
        )

        self.assertEqual(
            y.shape,
            (np.divide(16, self.slice_size[0]), np.divide(16, self.slice_size[1])),
            f"y's shape is not the right size. Current: f{y.shape}, Expected: (4,4)",
        )

    def test_getitem_with_data(self):
        dataset = self.dataset
        for subtile_index in range(len(dataset)):
            X, y = dataset[subtile_index]
            self.assertIsInstance(X, torch.Tensor)
            self.assertEqual(
                X.shape,
                (
                    # viirs: 9 dates 1 band
                    # s1: 4 dates 2 bands
                    # s2: 4 dates 12 bands
                    # landsat: 3 dates 11 bands
                    ((9 * 1) + (4 * 2) + (4 * 12) + (3 * 11)),
                    np.divide(800, self.slice_size[0]),
                    np.divide(800, self.slice_size[1]),
                ),
            )

            self.assertEqual(
                y.shape,
                (
                    np.divide(16, self.slice_size[0]),
                    np.divide(16, self.slice_size[1]),
                ),
            )


if __name__ == "__main__":
    unittest.main()
