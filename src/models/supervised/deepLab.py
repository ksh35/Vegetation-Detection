from torchvision.models.segmentation import fcn_resnet101
import torch
from torch import nn
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class DeepLab(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=50, **kwargs):
        """
        Loads the DeepLab model from torch hub,
        then replaces the first and last layer of the network
        in order to adapt it to our current problem, 
        the first convolution of the DeepLab must be changed
        to an input_channels -> 64 Conv2d with (7,7) kernel size,
        (2,2) stride, (3,3) padding and no bias.

        The last layer must be changed to be a 256 -> output_channels
        conv2d layer, with (1,1) kernel size and (1,1) stride. 


        
        Input:
            input_channels: number of input channels of the image
            of shape (batch, input_channels, width, height)
            output_channels: number of output channels of prediction,
            prediction is shape 4x400x400
        """
        super(DeepLab, self).__init__()

        # save in_channels and out_channels to self
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # use torch.hub to load 'pytorch/vision', 'fcn_resnet101', make sure to use pretrained=True
        # save it to self.model
        self.model = torch.hub.load(repo_or_dir='pytorch/vision', model='deeplabv3_resnet101', pretrained=True)
        
        # change self.model.backbone.conv1 to use in_channels as input
        self.model.backbone.conv1 = nn.Conv2d(in_channels=in_channels, out_channels = 64, kernel_size = (7,7), stride = (2,2), padding = (3,3), bias = False)
        # change self.model.classifier[-1] to use out_channels as output
        self.model.classifier[-1] = nn.Conv2d(in_channels = 256, out_channels=out_channels, kernel_size= (1,1), stride = (1, 1))
        # # create a final pooling layer that's a maxpool2d, of kernel size scale_factor
        # self.pool = nn.MaxPool2d(kernel_size=scale_factor)
        
    def forward(self, x):
        """
        Runs predictions on the DeepLab model
        Input:
            x: image to run a prediction of, of shape
            (batch, self.input_channels, width, height)
        Output:
            pred_y: predicted labels of size
        """
        # run x through self.model
        x = self.model(x)
        y_pred = x["out"]
        return y_pred