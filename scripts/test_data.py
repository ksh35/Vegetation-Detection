import sys
from argparse import ArgumentParser
from pathlib import Path
import xarray as xr
import numpy as np
from sklearn.metrics import f1_score
sys.path.append(".")
from src.esd_data.datamodule import ESDDataModule
from src.models.supervised.satellite_module import ESDSegmentation
from src.utilities import ESDConfig
import matplotlib.pyplot as plt
import torch
def main(options):
    options.processed_dir = Path(options.processed_dir)
    options.results_dir = Path(options.results_dir)
    datamodule = ESDDataModule(
        processed_dir=options.processed_dir,
        raw_dir=options.raw_dir,
        batch_size=options.batch_size,
        seed=options.seed,
        selected_bands=options.selected_bands,
        slice_size=options.slice_size,
    )
    model = ESDSegmentation.load_from_checkpoint(options.model_path)
    model.eval()
    datamodule.prepare_data()
    subtiles = list((options.processed_dir / "Train" / "subtiles").glob("Tile*")) + list((options.processed_dir / "Val" / "subtiles").glob("Tile*"))
    print("Last Processing part (no timer)")
    #Thresholds
    map = np.vectorize(lambda val: 0 if val < 0.2 else (1 if val < 0.4 else (2 if val < 0.6 else 3)))
    #Processing Data, saving only data need (RGB Bands and NDVI ground truth)
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

    f1_total = 0
    count = 0
    accuracy_total = 0
    subtiles = list((options.processed_dir / "Train" / "subtiles").glob("Tile*")) + list((options.processed_dir / "Val" / "subtiles").glob("Tile*"))
    #Plotting and Predicting
    for file in subtiles:
        for dir in file.glob("*"):
            if (dir / "landsat_rgb.nc").exists():
                landsat = xr.open_dataarray(dir / "landsat_rgb.nc")
                landsat = torch.tensor(landsat.values)
                landsat = landsat.unsqueeze(0)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                landsat = landsat.to(device)
                output = model(landsat)
                output = output.squeeze().cpu().detach().numpy()
                output = np.argmax(output, axis=0)
                #plots rgb, ground truth, and output
                fig, axs = plt.subplots(1, 3)
                axs[0].imshow(landsat.squeeze().permute(1, 2, 0).cpu().numpy())
                axs[0].set_title("RGB Image")
                ndvi = xr.open_dataarray(dir / "landsat_ndvi.nc").astype(int)
                ndvi_plot = axs[1].imshow(ndvi, cmap="viridis")
                axs[1].set_title("Ground Truth Vegetation")
                output_plot = axs[2].imshow(output.astype(int), cmap="viridis")
                axs[2].set_title("Predicted Vegetation") 
                plt.tight_layout()
                plt.savefig(options.results_dir / f"{file.name}_{dir.name}_output.png")
                plt.close()
                f1_total += f1_score(ndvi.values.flatten(), output.flatten(), average='macro')
                count += 1
                accuracy_total += np.mean(ndvi.values == output)
    print(f"Average Accuracy: {accuracy_total / count}")
    print(f"Average F1 Score: {f1_total / count}")
    

if __name__ == "__main__":
    config = ESDConfig()
    parser = ArgumentParser()
    root = Path.cwd()
    parser.add_argument(
        "--model_path", type=str, help="Model path.", default=config.model_path
    )
    parser.add_argument(
        "--raw_dir", type=str, default=root / "data" / "raw" / "Test", help="Path to raw directory"
    )
    parser.add_argument(
        "-p", "--processed_dir", type=str, default=root / "data" / "processed_test", help="."
    )
    parser.add_argument(
        "--results_dir", type=str, default=config.results_dir, help="Results dir"
    )
    main(ESDConfig(**parser.parse_args().__dict__))