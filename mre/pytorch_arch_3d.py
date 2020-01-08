# https://github.com/milesial/Pytorch-UNet
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from tensorboardX import SummaryWriter
from mre.CoordConv import CoordConv


def double_conv3d(in_channels, out_channels, kernel_size=(3, 5, 5)):
    '''Function for defining a standard double conv operation.'''

    padding = tuple([int((k-1)/2) for k in kernel_size])
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                  padding=padding),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size,
                  padding=padding),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True)
    )


def double_conv3d_depth(in_channels, out_channels, kernel_size=(3, 5, 5)):
    '''Function for defining a standard double conv operation.'''

    padding = tuple([int((k-1)/2) for k in kernel_size])
    return nn.Sequential(
        nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size,
                  padding=padding, groups=in_channels),
        nn.BatchNorm3d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1),
                  padding=(0, 0, 0)),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True)
    )


def double_conv3d_layer(in_channels, out_channels, kernel_size=(3, 5, 5), layer_size=None):
    '''Function for defining a standard double conv operation.'''

    padding = tuple([int((k-1)/2) for k in kernel_size])
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                  padding=padding),
        nn.LayerNorm([out_channels, layer_size[0], layer_size[1], layer_size[2]]),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size,
                  padding=padding),
        nn.LayerNorm([out_channels, layer_size[0], layer_size[1], layer_size[2]]),
        nn.ReLU(inplace=True)
    )


def down_layer3d(in_channels, out_channels, kernel_size=(3, 5, 5)):
    '''Simple down layer: maxpool then double conv'''
    return nn.Sequential(
        # nn.MaxPool3d((1, 2, 2), padding=(0, 0, 0)), # D2.5
        nn.MaxPool3d((2, 2, 2), padding=(0, 0, 0)),
        double_conv3d(in_channels, out_channels, kernel_size)
    )


def down_layer3d_depth(in_channels, out_channels, kernel_size=(3, 5, 5)):
    '''Simple down layer: maxpool then double conv'''
    return nn.Sequential(
        # nn.MaxPool3d((1, 2, 2), padding=(0, 0, 0)), # D2.5
        nn.MaxPool3d((2, 2, 2), padding=(0, 0, 0)),
        double_conv3d_depth(in_channels, out_channels, kernel_size)
    )


def down_layer3d_layer(in_channels, out_channels, kernel_size=(3, 5, 5), layer_size=None):
    '''Simple down layer: maxpool then double conv'''
    return nn.Sequential(
        # nn.MaxPool3d((1, 2, 2), padding=(0, 0, 0)), # D2.5
        nn.MaxPool3d((2, 2, 2), padding=(0, 0, 0)),
        double_conv3d_layer(in_channels, out_channels, kernel_size, layer_size)
    )


class up_layer3d(nn.Module):
    def __init__(self, in_channels, out_channels, channel_growth=True, kernel_size=(1, 2, 2)):
        '''Up layers require a class instead of a function in order to define a forward function
        that takes two inputs instead of 1 (for concat)'''
        super().__init__()

        # Note: may have to revist upsampling options (ConvTranspose3D might have padding issues?)
        stride = kernel_size
        if channel_growth:
            self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size,
                                               stride=stride)
        else:
            self.upsample = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=kernel_size,
                                               stride=stride)

        self.dconv = double_conv3d(in_channels, out_channels)

    def forward(self, x1, x2):

        x1 = self.upsample(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.dconv(x)


class up_layer3d_depth(nn.Module):
    def __init__(self, in_channels, out_channels, channel_growth=True, kernel_size=(1, 2, 2)):
        '''Up layers require a class instead of a function in order to define a forward function
        that takes two inputs instead of 1 (for concat)'''
        super().__init__()

        # Note: may have to revist upsampling options (ConvTranspose3D might have padding issues?)
        stride = kernel_size
        if channel_growth:
            self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size,
                                               stride=stride, groups=out_channels)
        else:
            self.upsample = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=kernel_size,
                                               stride=stride, groups=out_channels)

        self.dconv = double_conv3d_depth(in_channels, out_channels)

    def forward(self, x1, x2):

        x1 = self.upsample(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.dconv(x)


class GeneralUNet3D(nn.Module):
    def __init__(self, n_layers, in_channels, out_channels_init, out_channels_final,
                 channel_growth, coord_conv, transfer_layer=False, depth=False):
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

        self.layer_size_init = np.asarray((32, 256, 256))
        self.transfer_layer = transfer_layer
        if self.transfer_layer:
            raise ValueError('transfer not implemented')

        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        # self.in_layer = double_conv3d(in_channels, out_channels_init, kernel_size=(1, 1, 1))
        if depth:
            self.in_layer = double_conv3d_depth(in_channels, out_channels_init,
                                                kernel_size=(1, 1, 1))
        else:
            self.in_layer = double_conv3d(in_channels, out_channels_init, kernel_size=(1, 1, 1))
        self.out_layer = nn.Conv3d(out_channels_init, out_channels_final, 1)
        self.maxpool = nn.MaxPool3d(2)

        if channel_growth:
            for i in range(n_layers):
                down_kernel = (3, 5, 5)
                # up_kernel = (1, 2, 2) # D2.5
                up_kernel = (2, 2, 2)
                # if i == 0:
                #     up_kernel = (1, 1, 1)
                if 2 <= i <= 5:
                    down_kernel = (3, 3, 3)
                # Double number of channels for each down layer
                # self.down_layers.append(
                #     down_layer3d(out_channels_init*(2**i),
                #                  out_channels_init*(2**(i+1)), kernel_size=down_kernel)
                # )
                if depth:
                    self.down_layers.append(
                        down_layer3d_depth(out_channels_init*(2**i),
                                           out_channels_init*(2**(i+1)), kernel_size=down_kernel)
                    )
                    self.up_layers.append(
                        up_layer3d_depth(out_channels_init*(2**(i+1)),
                                         out_channels_init*(2**i), kernel_size=up_kernel)
                    )
                else:
                    self.down_layers.append(
                        down_layer3d(out_channels_init*(2**i),
                                     out_channels_init*(2**(i+1)), kernel_size=down_kernel)
                    )
                    self.up_layers.append(
                        up_layer3d(out_channels_init*(2**(i+1)),
                                   out_channels_init*(2**i), kernel_size=up_kernel)
                    )
                # self.down_layers.append(
                #     down_layer3d_layer(out_channels_init*(2**i),
                #                        out_channels_init*(2**(i+1)), kernel_size=down_kernel,
                #                        layer_size=self.layer_size_init//(2**(i+1)))
                # )
                # Quarter number of channels for each up layer (due to concats)
        else:
            for i in range(n_layers):
                # Keep number of channels const for down layer
                self.down_layers.append(
                    down_layer3d(out_channels_init,
                                 out_channels_init)
                )
                # Half number of channels for each up layer (due to concats)
                self.up_layers.append(
                    up_layer3d(out_channels_init*2,
                               out_channels_init, channel_growth)
                )

        # Reverse the order of up layers for easier iteration
        self.up_layers = self.up_layers[::-1]

    def forward(self, x):
        if self.transfer_layer:
            x = self.pretrained(x)
        skip_connects = []
        # Generate the input layer and save for skip connection
        # import pdb; pdb.set_trace()
        skip_connects.append(self.in_layer(x))
        # Generate the down-sampling layers and save for skip connections
        for down in self.down_layers:
            skip_connects.append(down(skip_connects[-1]))

        # revese skip connects for easier iteration
        skip_connects = skip_connects[::-1]
        x = skip_connects[0]
        # Generate the up-sampling layers and include skip connections
        for i, up in enumerate(self.up_layers):
            x = up(x, skip_connects[i+1])
        # Return the final layer
        return self.out_layer(x)
