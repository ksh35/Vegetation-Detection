#IMPORTANT
#pip install image if you want to run custom images (using this file)
from pathlib import Path
import xarray as xr
import sys
from argparse import ArgumentParser
sys.path.append(".")
from src.models.supervised.satellite_module import ESDSegmentation
from src.esd_data.datamodule import ESDDataModule
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


#turn jpg into 3 band rgb array (no ndvi, just turn into rgb)
def main(input_image_path, model_path):
    img = Image.open(input_image_path)
    img = img.resize((400, 400))
    #turn into 3 band rgb array
    img = np.array(img)
    img = img / 255
    img = img.transpose(2, 0, 1)

    #predicts the image
    img = torch.tensor(img)
    img = img.unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = img.to(device)
    model = ESDSegmentation.load_from_checkpoint(model_path)
    model.eval()
    output = model(img)
    output = output.squeeze().cpu().detach().numpy()
    output = np.argmax(output, axis=0)
    #plot the images in one file
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img.squeeze().permute(1, 2, 0).cpu().numpy())
    axs[0].set_title("RGB Image")
    axs[1].imshow(output.astype(int), cmap="viridis")
    axs[1].set_title("Predicted Vegetation Quality (NDVI)") 
    plt.savefig("custom_prediction.png")


if __name__ == "__main__":
    parser = ArgumentParser()
    root = Path.cwd()
    parser.add_argument(
        "--model_path", type=str, help="Model path.", default="models/DeepLab/last.ckpt"
    )
    parser.add_argument(
        "--image_path", type=str, default="custom_image.png", help="Image file path"
    )
    args = parser.parse_args()
    main(Path(args.image_path), Path(args.model_path))






