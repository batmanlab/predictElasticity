# https://github.com/milesial/Pytorch-UNet
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


def double_conv(in_channels, out_channels, coord_conv=False, kernel=3):
    '''Function for defining a standard double conv operation.  Additional option to replace first
    conv2d with CoordConv.'''

    if kernel >= 3:
        dilation = 2
    else:
        dilation = 1
    padding1 = int((kernel-1)/2)
    padding2 = int(((kernel*dilation-1)-1)/2)
    if coord_conv:
        first_2dconv = CoordConv(in_channels, out_channels, False, kernel_size=kernel,
                                 padding=padding1)
    else:
        first_2dconv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=padding2,
                                 dilation=dilation)

    return nn.Sequential(
        first_2dconv,
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=kernel, padding=padding1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def down_layer(in_channels, out_channels, kernel=3):
    '''Simple down layer: maxpool then double conv'''
    return nn.Sequential(
        nn.MaxPool2d(2),
        double_conv(in_channels, out_channels, kernel=kernel)
    )


def down_layer3d(in_channels, out_channels, kernel_size=(3, 5, 5)):
    '''Simple down layer: maxpool then double conv'''
    return nn.Sequential(
        # nn.MaxPool3d((1, 2, 2), padding=(0, 0, 0)), # D2.5
        nn.MaxPool3d((2, 2, 2), padding=(0, 0, 0)),
        double_conv3d(in_channels, out_channels, kernel_size)
    )


class up_layer(nn.Module):
    def __init__(self, in_channels, out_channels, channel_growth=True, kernel_size=2):
        '''Up layers require a class instead of a function in order to define a forward function
        that takes two inputs instead of 1 (for concat)'''
        super().__init__()
        stride = kernel_size

        # Note: may have to revist upsampling options (ConvTranspose2D might have padding issues?)
        if channel_growth:
            self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                                               stride=stride)
        else:
            self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=kernel_size,
                                               stride=stride)

        self.dconv = double_conv(in_channels, out_channels)

    def forward(self, x1, x2):

        x1 = self.upsample(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.dconv(x)


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


class GeneralUNet2D(nn.Module):
    def __init__(self, n_layers, in_channels, out_channels_init, out_channels_final,
                 channel_growth, coord_conv, transfer_layer=False):
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

        self.transfer_layer = transfer_layer
        if self.transfer_layer:
            self.pretrained = PretrainedModel('resnet50')

        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.in_layer = double_conv(in_channels, out_channels_init, coord_conv)
        self.out_layer = nn.Conv2d(out_channels_init, out_channels_final, 1)
        self.maxpool = nn.MaxPool2d(2)

        if channel_growth:
            for i in range(n_layers):
                # Double number of channels for each down layer
                up_kernel = 2
                if i == 0:
                    kernel = 7
                    up_kernel = 1
                elif i == 1:
                    kernel = 5
                    # up_kernel = 1
                elif 2 <= i <= 5:
                    kernel = 3
                else:
                    kernel = 1
                self.down_layers.append(
                    down_layer(out_channels_init*(2**i),
                               out_channels_init*(2**(i+1)), kernel)
                )
                # Quarter number of channels for each up layer (due to concats)
                self.up_layers.append(
                    up_layer(out_channels_init*(2**(i+1)),
                             out_channels_init*(2**i), up_kernel)
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
        if self.transfer_layer:
            x = self.pretrained(x)
        skip_connects = []
        # Generate the input layer and save for skip connection
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


class GeneralUNet3D(nn.Module):
    def __init__(self, n_layers, in_channels, out_channels_init, out_channels_final,
                 channel_growth, coord_conv, transfer_layer=False):
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

        self.transfer_layer = transfer_layer
        if self.transfer_layer:
            self.pretrained = PretrainedModel('resnet50')

        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.in_layer = double_conv3d(in_channels, out_channels_init, kernel_size=(1, 1, 1))
        self.out_layer = nn.Conv3d(out_channels_init, out_channels_final, 1)
        self.maxpool = nn.MaxPool3d(2)

        if channel_growth:
            for i in range(n_layers):
                down_kernel = (3, 5, 5)
                # up_kernel = (1, 2, 2) # D2.5
                up_kernel = (2, 2, 2)
                # if i == 0:
                # up_kernel = (1, 1, 1)
                if 2 <= i <= 5:
                    down_kernel = (3, 3, 3)
                # Double number of channels for each down layer
                self.down_layers.append(
                    down_layer3d(out_channels_init*(2**i),
                                 out_channels_init*(2**(i+1)), kernel_size=down_kernel)
                )
                # Quarter number of channels for each up layer (due to concats)
                self.up_layers.append(
                    up_layer3d(out_channels_init*(2**(i+1)),
                               out_channels_init*(2**i), kernel_size=up_kernel)
                )
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


class PretrainedModel(nn.Module):
    def __init__(self, arch_name):
        super().__init__()
        self.model_trans = models.resnet50(pretrained=True)
        self.transfer_layer1 = nn.Sequential(*list(self.model_trans.children())[0:3])
        self.transfer_layer2 = nn.Sequential(*list(self.model_trans.children())[3:5])
        for param in self.transfer_layer1:
            param.requires_grad = True
        for param in self.transfer_layer2:
            param.requires_grad = True

    def forward(self, x):
        t_layer1 = self.transfer_layer1(x)
        t_layer2 = self.transfer_layer2(t_layer1)
        r_layer1 = torch.reshape(t_layer1, (t_layer1.shape[0], -1, 224, 224))
        r_layer2 = torch.reshape(t_layer2, (t_layer2.shape[0], -1, 224, 224))
        return torch.cat([r_layer1, r_layer2], dim=1)
