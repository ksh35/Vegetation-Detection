from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List

import xarray as xr


# given
class SatelliteType(Enum):
    VIIRS_MAX_PROJ = "viirs_max_projection"
    VIIRS = "viirs"
    S1 = "sentinel1"
    S2 = "sentinel2"
    LANDSAT = "landsat"
    GT = "gt"


ROOT = Path.cwd()
PROJ_NAME = "CS175-spring-2024"
MODEL = "FCNResnetTransfer"  # default, valid values are ["UNet", "SegmentationCNN", "FCNResnetTransfer"] # fmt: skip


@dataclass
class ESDConfig:
    processed_dir: Path = ROOT / "data" / "processed"
    raw_dir: Path = ROOT / "data" / "raw" / "Train"
    results_dir: Path = ROOT / "data" / "predictions" / MODEL
    selected_bands = {
        #SatelliteType.VIIRS: ["0"],
        #SatelliteType.S1: ["VV", "VH"],
        # SatelliteType.S2: [
        #     "12",
        #     "11",
        #     "09",
        #     "8A",
        #     "08",
        #     "07",
        #     "06",
        #     "05",
        #     "04",
        #     "03",
        #     "02",
        #     "01",
        # ],
        SatelliteType.LANDSAT: [
            "11",
            "10",
            "9",
            "8",
            "7",
            "6",
            "5",
            "4",
            "3",
            "2",
            "1",
        ],
        #SatelliteType.VIIRS_MAX_PROJ: ["0"],
    }

    accelerator: str = "cpu"
    batch_size: int = 2
    depth: int = 2
    devices: int = 1
    embedding_size: int = 64
    in_channels: int = 99  # num_dates * num_bands
    kernel_size: int = 3
    learning_rate: float = 1e-3
    max_epochs: int = 1
    model_path: Path = ROOT / "models" / MODEL / "last.ckpt"
    model_type: str = MODEL
    n_encoders: int = 2
    num_workers: int = 15
    out_channels: int = 4  # 4 out channels for our 4 classes
    pool_sizes: str = "5,5,2"
    seed: int = 12378921
    slice_size: tuple = (2, 2)
    wandb_run_name: str | None = None


# given
def get_satellite_dataset_size(
    data_set: xr.Dataset, dims: List[str] = ["date", "band", "height", "width"]
):
    """
    Gets the shape of a dataset

    Parameters
    ----------
    data_set : xr.Dataset
        A satellite dataset
    dims: List[str]
        A list of dimensions of the data_set data_arrays
    Returns
    -------
    Tuple:
        Shape of the data_set, default is (date, band, height, width)
    """
    return tuple(data_set.sizes[d] for d in dims)
