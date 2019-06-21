# https://github.com/milesial/Pytorch-UNet
import torch
import torch.nn as nn
from torchvision import models
from tensorboardX import SummaryWriter
from mre.CoordConv import CoordConv


def double_conv(in_channels, out_channels, coord_conv=False):
    '''Function for defining a standard double conv operation.  Additional option to replace first
    conv2d with CoordConv.'''

    if coord_conv:
        first_2dconv = CoordConv(in_channels, out_channels, False, kernel_size=3, padding=1)
    else:
        first_2dconv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    return nn.Sequential(
        first_2dconv,
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def down_layer(in_channels, out_channels):
    '''Simple down layer: maxpool then double conv'''
    return nn.Sequential(
        nn.MaxPool2d(2),
        double_conv(in_channels, out_channels)
    )


class up_layer(nn.Module):
    def __init__(self, in_channels, out_channels, channel_growth=True):
        '''Up layers require a class instead of a function in order to define a forward function
        that takes two inputs instead of 1 (for concat)'''
        super().__init__()

        # Note: may have to revist upsampling options (ConvTranspose2D might have padding issues?)
        if channel_growth:
            self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)

        self.dconv = double_conv(in_channels, out_channels)

    def forward(self, x1, x2):

        x1 = self.upsample(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.dconv(x)


class GeneralUNet(nn.Module):
    def __init__(self, n_layers, in_channels, out_channels_init, out_channels_final,
                 channel_growth, coord_conv):
        '''Generalized UNet class, meant to be highly modular.  Allows for variable number of layers
        and other features, but is less human-readable as a result.

        Args:
            n_layers (int): Number of down (up) layers to generate.  Must be >=0 (input and output
                layers count as zeroth layer)
            in_channels (int): Number of input channels.  Typically 3 if original data is input, but
                could be much more if transfer learning used.
            out_channels_init (int): Number of channels produced in the first down layer.  If
                `channel_growth` is False, this defines total number of channels in every layer.
            out_channels_final (int): Number of channels produced in the last layer.  Typically 1
                for standard MRE regression.
            channel_growth (bool): If True, number of channels will double (half) with each down
                (up) layer.
            coord_conv (bool): If True, first down layer will use a coord_conv operation.
        '''
        super().__init__()

        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.in_layer = double_conv(in_channels, out_channels_init, coord_conv)
        self.out_layer = nn.Conv2d(out_channels_init, out_channels_final, 1)
        self.maxpool = nn.MaxPool2d(2)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.upsample = nn.ConvTranspose2d(

        if channel_growth:
            for i in range(n_layers):
                # Double number of channels for each down layer
                self.down_layers.append(
                    down_layer(out_channels_init*(2**i),
                               out_channels_init*(2**(i+1)))
                )
                # Quarter number of channels for each up layer (due to concats)
                self.up_layers.append(
                    up_layer(out_channels_init*(2**(i+1)),
                             out_channels_init*(2**i))
                )
        else:
            for i in range(n_layers):
                # Keep number of channels const for down layer
                self.down_layers.append(
                    down_layer(out_channels_init,
                               out_channels_init)
                )
                # Half number of channels for each up layer (due to concats)
                self.up_layers.append(
                    up_layer(out_channels_init*2,
                             out_channels_init, channel_growth)
                )

        # Reverse the order of up layers for easier iteration
        self.up_layers = self.up_layers[::-1]

    def forward(self, x):
        skip_connects = []
        skip_connects.append(self.in_layer(x))
        for down in self.down_layers:
            skip_connects.append(down(skip_connects[-1]))

        # revese skip connects for easier iteration
        skip_connects = skip_connects[::-1]
        x = skip_connects[0]
        for i, up in enumerate(self.up_layers):
            x = up(x, skip_connects[i+1])
        return self.out_layer(x)
