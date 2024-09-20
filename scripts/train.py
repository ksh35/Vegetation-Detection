import sys
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path
import pytorch_lightning as pl
import wandb
import xarray as xr
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)

sys.path.append(".")
from src.esd_data.datamodule import ESDDataModule
from src.models.supervised.satellite_module import ESDSegmentation
from src.utilities import ESDConfig, PROJ_NAME
import numpy as np
ROOT = Path.cwd()


def train(options: ESDConfig, accelerator: str):
    # initialize wandb
    # setup the wandb logger
    wandb_logger = WandbLogger(project="cs175-idk", config=options)
    # if wandb.run.sweep_id:
    #     wandb.run.tags = [f"Sweep {wandb.run.sweep_id}"]
    # initialize the datamodule

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
    #Processes the data, saving only RGB images and NDVI images
    map = np.vectorize(lambda val: 0 if val < 0.2 else (1 if val < 0.4 else (2 if val < 0.6 else 3)))

    for subtile in subtiles:
        for part in subtile.glob("*"):
            if((part / "landsat.nc").exists() == False):
                continue
            landsat = part / "landsat.nc"
            landsat = xr.open_dataarray(landsat).mean(dim="date")
            landsat_rgb = landsat.sel(band=["4", "3", "2"])
            landsat_ndvi= (landsat.sel(band="5") - landsat.sel(band="4")) / (landsat.sel(band="5") + landsat.sel(band="4")).squeeze()
            #Thresholding the data
            landsat_ndvi = xr.apply_ufunc(map, landsat_ndvi)
            #save new data
            landsat_rgb.to_netcdf(part / "landsat_rgb.nc")
            landsat_ndvi.to_netcdf(part / "landsat_ndvi.nc")
            landsat.close()
            (part / "landsat.nc").unlink()
            (part / "gt.nc").unlink()
        

    datamodule.setup("fit")

    # create a model params dict to initialize ESDSegmentation
    model_params = {}
    if options.model_type == "UNet":
        model_params = {
            "n_encoders": options.n_encoders,
            "embedding_size": options.embedding_size,
            "model_type": "UNet"
        }
    elif options.model_type == "FCNResnetTransfer":
        model_params = {
            "model_type": "FCNResnetTransfer"
        }
    elif options.model_type == "SegmentationCNN":
        model_params = {
            "depth": options.depth,
            "kernel_size": options.kernel_size,
            "pool_size": options.pool_sizes,
            "model_type": "SegmentationCNN"
        }
    elif options.model_type == "DeepLab":
        model_params = {
            "model_type": "DeepLab"
        }

    # note: different models have different parameters

    # initialize the ESDSegmentation model
    segmentation_model = ESDSegmentation(
        model_type=options.model_type,
        in_channels=3,
        out_channels=4,
        model_params=model_params,
        learning_rate=options.learning_rate,
    )
    # Use the following callbacks, they're provided for you,
    # but you may change some of the settings
    # ModelCheckpoint: saves intermediate results for the neural network
    # in case it crashes
    # LearningRateMonitor: logs the current learning rate on weights and biases
    # RichProgressBar: nicer looking progress bar (requires the rich package)
    # RichModelSummary: shows a summary of the model before training (requires rich)
    callbacks = [
        ModelCheckpoint(
            dirpath=ROOT / "models" / options.model_type,
            filename="{epoch}-{val_loss:.2f}-{other_metric:.2f}",
            save_top_k=0,
            save_last=True,
            verbose=True,
            monitor="val_loss",
            mode="min",
            every_n_train_steps=1000,
        ),
        LearningRateMonitor(),
        RichProgressBar(),
        RichModelSummary(max_depth=3),
    ]

    # initialize trainer, set accelerator, devices, number of nodes, logger
    # max epochs and callbacks
    trainer = pl.Trainer(
        accelerator = accelerator,
        devices = options.devices,
        num_nodes = 1,
        max_epochs= options.max_epochs,
        callbacks= callbacks,
        logger = wandb_logger
    )
    # run trainer.fit
    trainer.fit(segmentation_model, datamodule=datamodule)
    


if __name__ == "__main__":
    # load dataclass arguments from yml file

    config = ESDConfig()
    parser = ArgumentParser()

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
    parser.add_argument(
        "--device",
        type=str,
        help="Device to train on",
        default="cpu",
    )
    parse_args = parser.parse_args()
    accelerator = "gpu" if "gpu" in parse_args.device else "cpu"
    #remove device from parse args
    parse_args.__dict__.pop("device")
    config = ESDConfig(**parse_args.__dict__)

    train(config, accelerator)
