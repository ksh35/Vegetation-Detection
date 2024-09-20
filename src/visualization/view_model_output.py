from pathlib import Path
import xarray as xr
import sys
sys.path.append(".")
from src.models.supervised.satellite_module import ESDSegmentation
from src.esd_data.datamodule import ESDDataModule
import torch
import matplotlib.pyplot as plt
import numpy as np
input_image_path = Path("data/processed/Val/subtiles/Tile40/0_0/landsat_rgb.nc")
ndvi_image_path = Path("data/processed/Val/subtiles/Tile40/0_0/landsat_ndvi.nc")

model_path = Path("models/DeepLab/last.ckpt")

landsat = torch.tensor(xr.open_dataarray(input_image_path).values)
ndvi = xr.open_dataarray(ndvi_image_path).astype(int)


#predicts the image
landsat = landsat.unsqueeze(0)
landsat = landsat.cuda()
model = ESDSegmentation.load_from_checkpoint(model_path)
model.eval()
output = model(landsat)

#plots the images in one file
fig, axs = plt.subplots(1, 3)
axs[0].imshow(landsat.squeeze().permute(1, 2, 0).cpu().numpy())
axs[0].set_title("RGB Image")
axs[1].imshow(ndvi, cmap="viridis")
axs[1].set_title("Ground Truth NDVI")
output = output.squeeze().cpu().detach().numpy()
output = np.argmax(output, axis=0)
axs[2].imshow(output.astype(int), cmap="viridis")
axs[2].set_title("Predicted NDVI")
plt.savefig("output.png")

#computes accurracy between two numpy arrays:
def compute_accurracy(gt, pred):
    return np.sum(gt == pred) / gt.size

gt = ndvi.values
gt = gt.flatten()
output = output.flatten()
print(compute_accurracy(gt, output))