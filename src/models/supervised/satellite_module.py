import torch
import pytorch_lightning as pl
from torch.optim import Adam
from torch import nn
import torchmetrics
from src.models.supervised.segmentation_cnn import SegmentationCNN
from src.models.supervised.unet import UNet
from src.models.supervised.resnet_transfer import FCNResnetTransfer
from src.models.supervised.deepLab import DeepLab
import wandb


class ESDSegmentation(pl.LightningModule):
    def __init__(
            self,
            model_type,
            in_channels,
            out_channels,
            learning_rate=1e-3,
            model_params: dict = {},
    ):
        """
        Constructor for ESDSegmentation class.
        """
        # call the constructor of the parent class
        super().__init__()
        # use self.save_hyperparameters to ensure that the module will load

        new_dict = model_params.copy()
        new_dict["model_type"] = model_type
        new_dict["in_channels"] = in_channels
        new_dict["out_channels"] = out_channels
        self.save_hyperparameters(new_dict)

        # store in_channels and out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.learning_rate = learning_rate
        # if the model type is segmentation_cnn, initalize a unet as self.model
        if model_type == "SegmentationCNN":
            self.model = SegmentationCNN(in_channels=in_channels, out_channels=out_channels, **model_params)
        # if the model type is unet, initialize a unet as self.model
        elif model_type == "UNet":
            self.model = UNet(
                in_channels=in_channels, out_channels=out_channels, **model_params
            )
        # if the model type is fcn_resnet_transfer, initialize a fcn_resnet_transfer as self.model
        elif model_type == "FCNResnetTransfer":
            self.model = FCNResnetTransfer(
                in_channels=in_channels, out_channels=out_channels, **model_params
            )
        elif model_type == "DeepLab":
            self.model = DeepLab(
                in_channels=in_channels, out_channels=out_channels, **model_params
            )
        # initialize the accuracy metrics for the semantic segmentation task
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=out_channels)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=out_channels)
        self.f1 = torchmetrics.F1Score(num_classes=out_channels, task="multiclass")
        self.loss = nn.CrossEntropyLoss()

    def forward(self, X):
        # evaluate self.model
        X = X.to(torch.float32)
        X = self.model(X)
        return X

    def training_step(self, batch, batch_idx):
        # get sat_img and mask from batch
        sat_img, mask = batch

        # evaluate batch
        eval = self.forward(sat_img)
        # calculate cross entropy loss
        mask = mask.type(torch.long)
        loss = nn.CrossEntropyLoss()(eval, mask)

        # return loss
        return loss

    def validation_step(self, batch, batch_idx):
        # get sat_img and mask from batch
        sat_img, mask = batch
        # evaluate batch for validation
        eval = self.forward(sat_img)
        # get the class with the highest probability
        prob = torch.argmax(eval, dim=1)
        # evaluate each accuracy metric and log it in wandb

        acc_val = self.val_acc(eval, mask)
        f1 = self.f1(eval, mask)
        wandb.log({"val_f1": f1, "prob": prob, "acc_val": acc_val})

        # return validation loss
        mask = mask.type(torch.long)
        return self.loss(eval, mask)

    def configure_optimizers(self):
        # initialize optimizer
        optimizer = Adam(params=self.model.parameters(), lr=self.learning_rate)
        # return optimizer
        return optimizer
