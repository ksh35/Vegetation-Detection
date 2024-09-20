"""Legacy code from hw03 datamodule.py winter 2024"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pytorch_lightning as pl
import torch
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import xarray as xr
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from sklearn.model_selection import train_test_split
from torchvision import transforms as torchvision_transforms
from tqdm import tqdm

sys.path.append(".")
from src.esd_data.augmentations import (
    AddNoise,
    Blur,
    RandomHFlip,
    RandomVFlip,
    ToTensor,
)
from src.esd_data.dataset import ESDDataset
from src.preprocessing.file_utils import load_satellite
from src.preprocessing.preprocess_sat import (
    maxprojection_viirs,
    preprocess_landsat,
    preprocess_sentinel1,
    preprocess_sentinel2,
    preprocess_viirs,
)
from src.preprocessing.subtile import Subtile
from src.utilities import SatelliteType


# given
def collate_fn(batch):
    Xs = []
    ys = []
    for X, y in batch:
        Xs.append(X)
        ys.append(y)

    Xs = torch.stack(Xs)
    ys = torch.stack(ys)
    return Xs, ys


class ESDDataModule(pl.LightningDataModule):
    """
    The ESDDataModule class is designed to encapsulate data loading and processing logic for a model within the PyTorch Lightning framework.

    Attributes:
        processed_dir: Path to the directory containing processed data.
        raw_dir: Path to the directory containing raw data.
        batch_size: Batch size for data loaders.
        seed: Random seed for data processing.
        selected_bands: Dictionary mapping SatelliteType to a list of selected bands.
        slice_size: Tuple specifying the size for subtiling.
        train_size: Fraction of data allocated for training.
        transform_list: List of torchvision transforms applied to the data.

    Methods:
        load_and_preprocess(tile_dir: Path) -> Tuple[List[xr.DataArray], xr.DataArray]:
            Loads and preprocesses tile data located in tile_dir.

        prepare_data() -> None:
            Processes raw data by loading, splitting into train-test, and subtiling for training and validation.

        setup(stage: str) -> None:
            Sets up the training and validation datasets (self.train_dataset, self.val_dataset).

        train_dataloader() -> torch.utils.data.DataLoader:
            Creates and returns a DataLoader for the training dataset.

        val_dataloader() -> torch.utils.data.DataLoader:
            Creates and returns a DataLoader for the validation dataset.
    """

    # given
    def __init__(
        self,
        processed_dir: Path,
        raw_dir: Path,
        batch_size: int = 32,
        # might need to set it to 0
        num_workers: int = 0,
        seed: int = 12378921,
        selected_bands: Dict[SatelliteType, List[str]] = None,
        slice_size: Tuple[int, int] = (4, 4),
        train_size: float = 0.8,
        transform_list: list = [
            torchvision_transforms.RandomApply([AddNoise()], p=0.5),
            torchvision_transforms.RandomApply([Blur()], p=0.5),
            torchvision_transforms.RandomApply([RandomHFlip()], p=0.5),
            torchvision_transforms.RandomApply([RandomVFlip()], p=0.5),
            ToTensor(),
        ],
    ):
        super(ESDDataModule, self).__init__()
        self.processed_dir = processed_dir
        self.raw_dir = raw_dir
        self.batch_size = batch_size
        self.seed = seed
        self.selected_bands = selected_bands
        self.slice_size = slice_size
        self.train_size = train_size
        self.satellite_type_list = [key for key in self.selected_bands.keys()]
        self.transform = torchvision_transforms.transforms.Compose(transform_list)
        self.num_workers = num_workers
        self.train_dir = self.processed_dir / "Train"
        self.val_dir = self.processed_dir / "Val"

    # given
    def load_and_preprocess(
        self,
        tile_dir: Path,
    ) -> Tuple[List[xr.DataArray], xr.DataArray]:
        """
        Performs the preprocessing step: for a given tile located in tile_dir,
        loads the tif files and preprocesses them just like in homework 1.

        Input:
            tile_dir: Path
                Location of raw tile data
            satellite_type_list: List[SatelliteType]
                List of satellite types to process

        Output:


        """
        preprocess_functions = {
            SatelliteType.VIIRS: preprocess_viirs,
            SatelliteType.S1: preprocess_sentinel1,
            SatelliteType.S2: preprocess_sentinel2,
            SatelliteType.LANDSAT: preprocess_landsat,
        }
        satellite_type_list = self.satellite_type_list

        preprocessed_data_array_list = list()
        for satellite_type in satellite_type_list:
            if satellite_type != SatelliteType.VIIRS_MAX_PROJ:
                data_array = load_satellite(tile_dir, satellite_type)

                preprocessed_data_array_list.append(
                    preprocess_functions[satellite_type](
                        data_array.sel(band=self.selected_bands[satellite_type])
                    )
                )

        if SatelliteType.VIIRS_MAX_PROJ in satellite_type_list:
            preprocessed_data_array_list.append(
                maxprojection_viirs(load_satellite(tile_dir, SatelliteType.VIIRS))
            )

        return preprocessed_data_array_list, load_satellite(tile_dir, SatelliteType.GT)

    def prepare_data(self) -> None:
        """
        If the data has not been processed before (denoted by whether or not self.processed_dir is an existing directory),
        we will process it.

        The steps for processing are as follows:
            - load all 60 tiles
            - train test split the tiles
            - subtile and save the training split
            - subtile and save the testing split
        """
        # --- start here ---
        # if the processed_dir does not exist or its empty
        if not self.processed_dir.exists() or not list(self.processed_dir.iterdir()):
            # get the a list of the tile directories from the raw directory (use .glob() for simplicity)
            tile_dirs = list(self.raw_dir.glob("Tile*"))
            # call the train_test_split function.

            # The arrays will be the above list, the test_size will be 1 - train size, and the random_state will be the seed.
            # The output of this function will be a tuple, (tile directories for training the model, tile directories for validating the model).
            # Save the output into the two variables tile_dirs_train and tile_dirs_val.
            if self.train_size == 1:
                tile_dirs_train, tile_dirs_val = tile_dirs, []
            else:
                tile_dirs_train, tile_dirs_val = train_test_split(
                                tile_dirs,
                                test_size=1 - self.train_size,
                                random_state=self.seed)
            # We have now created the train test split. We are going to subtile and save these into
            # the self.train_dir and self.val_dir

            # iterate over the tile directories in tqdm(list(tile_dirs_train), desc="Processing train tiles")
            for tile_dir in tqdm(list(tile_dirs_train), desc="Processing train tiles"):
                # get the data array list and gt data array from load_and_preprocess
                data_array_list, gt_data_array = self.load_and_preprocess(tile_dir)

                # create a subtile, passing the data array list, gt data array, and the slice size
                subtile = Subtile(data_array_list, gt_data_array, self.slice_size)
                # save the subtile to the train_dir
                subtile.save(self.train_dir)

            # iterate over the tile directories in tqdm(list(tile_dirs_val), desc="Processing validation tiles"):
            for tile_dir in tqdm(
                list(tile_dirs_val), desc="Processing validation tiles"
            ):
                # get the data array list and gt data array from load_and_preprocess
                data_array_list, gt_data_array = self.load_and_preprocess(tile_dir)

                # create a subtile, passing the data array list, gt data array, and the slice size
                subtile = Subtile(data_array_list, gt_data_array, self.slice_size)
                # save the subtile to the val_dir
                subtile.save(self.val_dir)

    def setup(self, stage: str) -> None:
        """
        Create the self.train_dataset and self.val_dataset.
        """
        if stage == "fit":
            # --- start here ---
            # create the train ESDDataset (the processed_dir will be the train_dir)
            self.train_dataset = ESDDataset(
                self.train_dir,
                self.transform,
                self.satellite_type_list,
                self.slice_size,
            )
            # create the val ESDDataset (the processed_dir will be the val_dir)
            self.val_dataset = ESDDataset(
                self.val_dir, self.transform, self.satellite_type_list, self.slice_size
            )
        if stage == "test":
            self.test_dataset = ESDDataset(
                self.train_dir,
                transform=self.transform,
                satellite_type_list=self.satellite_type_list,
                slice_size=self.slice_size,
            )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Creates and returns a DataLoader with self.train_dataset
        """
        # create the torch.utils.data.Dataloader for the train_dataset, passing the batch size
        # and collate_fn
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Creates and returns a DataLoader with self.val_dataset
        """
        # create the torch.utils.data.Dataloader for the val_dataset, passing the batch size
        # and collate_fn
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:

        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
        )
