
import sys
from argparse import ArgumentParser
import xarray as xr
sys.path.append(".")
from src.esd_data.datamodule import ESDDataModule
from src.utilities import SatelliteType
import numpy as np
from src.utilities import ESDConfig, PROJ_NAME


def prepare_data(config: ESDConfig):
    datamodule = ESDDataModule(
        processed_dir=config.processed_dir,
        raw_dir=config.raw_dir,
        batch_size=config.batch_size,
        seed=config.seed,
        selected_bands=config.selected_bands,
        slice_size=config.slice_size,
    )

    datamodule.prepare_data()
    subtiles = list((config.processed_dir / "Train" / "subtiles").glob("Tile*")) + list((config.processed_dir / "Val" / "subtiles").glob("Tile*"))
    print("Last Processing part (no timer)")
    #Thresholds
    map = np.vectorize(lambda val: 0 if val < 0.2 else (1 if val < 0.4 else (2 if val < 0.6 else 3)))
    #Makes the landsat_rgb and landsat_ndvi files. Gets the mean image across the dates  
    for subtile in subtiles:
        for part in subtile.glob("*"):
            if((part / "landsat.nc").exists() == False):
                continue
            landsat = part / "landsat.nc"
            landsat = xr.open_dataarray(landsat).mean(dim="date")
            landsat_rgb = landsat.sel(band=["4", "3", "2"])
            landsat_ndvi= (landsat.sel(band="5") - landsat.sel(band="4")) / (landsat.sel(band="5") + landsat.sel(band="4")).squeeze()

            landsat_ndvi = xr.apply_ufunc(map, landsat_ndvi)
            landsat_rgb.to_netcdf(part / "landsat_rgb.nc")
            landsat_ndvi.to_netcdf(part / "landsat_ndvi.nc")
            landsat.close()
            (part / "landsat.nc").unlink()
            (part / "gt.nc").unlink()
        

    datamodule.setup("fit")



if __name__ == "__main__":
    config = ESDConfig()
    parser = ArgumentParser()


    parse_args = parser.parse_args()

    parser.add_argument(
        "--model_type",
        type=str,
        help="The model to initialize.",
        default=config.model_type,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="The learning rate for training model",
        default=config.learning_rate,
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        help="Number of epochs to train for.",
        default=config.max_epochs,
    )
    parser.add_argument(
        "--raw_dir", type=str, default=config.raw_dir, help="Path to raw directory"
    )
    parser.add_argument(
        "-p", "--processed_dir", type=str, default=config.processed_dir, help="."
    )

    parser.add_argument(
        "--in_channels",
        type=int,
        default=config.in_channels,
        help="Number of input channels",
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=config.out_channels,
        help="Number of output channels",
    )
    parser.add_argument(
        "--depth",
        type=int,
        help="Depth of the encoders (CNN only)",
        default=config.depth,
    )
    parser.add_argument(
        "--n_encoders",
        type=int,
        help="Number of encoders (Unet only)",
        default=config.n_encoders,
    )
    parser.add_argument(
        "--embedding_size",
        type=int,
        help="Embedding size of the neural network (CNN/Unet)",
        default=config.embedding_size,
    )
    parser.add_argument(
        "--pool_sizes",
        help="A comma separated list of pool_sizes (CNN only)",
        type=str,
        default=config.pool_sizes,
    )
    parser.add_argument(
        "--kernel_size",
        help="Kernel size of the convolutions",
        type=int,
        default=config.kernel_size,
    )

    config = ESDConfig(**parse_args.__dict__)

    prepare_data(config)

