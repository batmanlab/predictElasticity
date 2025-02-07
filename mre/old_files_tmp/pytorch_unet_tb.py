# https://github.com/usuyama/pytorch-unet
import torch
import torch.nn as nn
from torchvision import models
from tensorboardX import SummaryWriter
from mre.CoordConv import CoordConv


def double_cord_conv(in_channels, out_channels):
    return nn.Sequential(
        CoordConv(in_channels, out_channels, False, kernel_size=3, padding=1),
        # nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):

    def __init__(self, n_class, cap=16, coord_conv=True):
        super().__init__()

        if coord_conv:
            self.dconv_down1 = double_cord_conv(3, cap)
        else:
            self.dconv_down1 = double_conv(3, cap)
        self.dconv_down2 = double_conv(cap, cap)
        self.dconv_down3 = double_conv(cap, cap)
        self.dconv_down4 = double_conv(cap, cap)
        self.dconv_down5 = double_conv(cap, cap)
        self.dconv_down6 = double_conv(cap, cap)
        self.dconv_down7 = double_conv(cap, cap)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up6 = double_conv(cap + cap, cap)
        self.dconv_up5 = double_conv(cap + cap, cap)
        self.dconv_up4 = double_conv(cap + cap, cap)
        self.dconv_up3 = double_conv(cap + cap, cap)
        self.dconv_up2 = double_conv(cap + cap, cap)
        self.dconv_up1 = double_conv(cap + cap, cap)

        self.conv_last = nn.Conv2d(cap, n_class, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)

        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)

        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)

        x = self.maxpool(conv3)
        conv4 = self.dconv_down4(x)

        x = self.maxpool(conv4)
        conv5 = self.dconv_down5(x)

        x = self.maxpool(conv5)
        conv6 = self.dconv_down6(x)

        x = self.maxpool(conv6)
        x = self.dconv_down7(x)

        x = self.upsample(x)
        x = torch.cat([x, conv6], dim=1)
        x = self.dconv_up6(x)

        x = self.upsample(x)
        x = torch.cat([x, conv5], dim=1)
        x = self.dconv_up5(x)

        x = self.upsample(x)
        x = torch.cat([x, conv4], dim=1)
        x = self.dconv_up4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)

        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)

        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class ResNetUNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()

        # Use ResNet18 as the encoder with the pretrained weights
        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 256, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(256, 256, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 512, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(512, 512, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 1024, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(1024, 512, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 2048, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(2048, 1024, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(512 + 1024, 512, 3, 1)
        self.conv_up2 = convrelu(512 + 512, 512, 3, 1)
        self.conv_up1 = convrelu(256 + 512, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        # Upsample the last/bottom layer
        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        # Create the shortcut from the encoder
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out


class UNet_Partial(nn.Module):

    def __init__(self, n_class, cap=16, stage='late'):
        super().__init__()

        self.dconv_down1 = double_conv(3, cap)
        self.dconv_down2 = double_conv(cap, cap)
        self.dconv_down3 = double_conv(cap, cap)
        self.dconv_down4 = double_conv(cap, cap)
        self.dconv_down5 = double_conv(cap, cap)
        self.dconv_down6 = double_conv(cap, cap)
        self.dconv_down7 = double_conv(cap, cap)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up6 = double_conv(cap + cap, cap)
        self.dconv_up5 = double_conv(cap + cap, cap)
        self.dconv_up4 = double_conv(cap + cap, cap)
        self.dconv_up3 = double_conv(cap + cap, cap)
        self.dconv_up2 = double_conv(cap + cap, cap)
        self.dconv_up1 = double_conv(cap + cap, cap)

        self.conv_last = nn.Conv2d(cap, n_class, 1)
        self.stage = stage

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        if self.stage == 'early':
            return x

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)

        conv5 = self.dconv_down5(x)
        x = self.maxpool(conv5)
        if self.stage == 'mid':
            return x

        conv6 = self.dconv_down6(x)
        x = self.maxpool(conv6)

        x = self.dconv_down7(x)

        x = self.upsample(x)
        x = torch.cat([x, conv6], dim=1)

        x = self.dconv_up6(x)
        x = self.upsample(x)
        x = torch.cat([x, conv5], dim=1)

        x = self.dconv_up5(x)
        x = self.upsample(x)
        x = torch.cat([x, conv4], dim=1)

        x = self.dconv_up4(x)
        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        if self.stage == 'late':
            return x
        # x = torch.cat([x, conv2], dim=1)

        # x = self.dconv_up2(x)
        # x = self.upsample(x)
        # x = torch.cat([x, conv1], dim=1)

        # x = self.dconv_up1(x)

        # out = self.conv_last(x)

        return x


class UNet_Transfer(nn.Module):
    def __init__(self, n_class, cap=16, coord_conv=True):
        super().__init__()
        self.model_trans = models.resnet50(pretrained=True)
        self.transfer_layer1 = nn.Sequential(*list(self.model_trans.children())[0:3])
        self.transfer_layer2 = nn.Sequential(*list(self.model_trans.children())[3:5])
        if coord_conv:
            self.dconv_down1 = double_cord_conv(256, cap)
        else:
            self.dconv_down1 = double_conv(256, cap)
        self.dconv_down2 = double_conv(cap, cap)
        self.dconv_down3 = double_conv(cap, cap)
        self.dconv_down4 = double_conv(cap, cap)
        self.dconv_down5 = double_conv(cap, cap)
        self.dconv_down6 = double_conv(cap, cap)
        self.dconv_down7 = double_conv(cap, cap)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up6 = double_conv(cap + cap, cap)
        self.dconv_up5 = double_conv(cap + 256, 256)
        self.dconv_up4 = double_conv(cap + cap, cap)
        self.dconv_up3 = double_conv(cap + cap, cap)
        self.dconv_up2 = double_conv(cap + cap, cap)
        self.dconv_up1 = double_conv(256 + 64, cap)

        self.conv_last = nn.Conv2d(cap, n_class, 1)

    def forward(self, x):
        t_layer1 = self.transfer_layer1(x)
        t_layer2 = self.transfer_layer2(t_layer1)
        x = self.maxpool(t_layer2)

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        x = self.dconv_down2(x)
        # x = self.maxpool(conv2)

        # conv3 = self.dconv_down3(x)
        # x = self.maxpool(conv3)

        # x = self.dconv_down4(x)
        # x = self.maxpool(conv4)

        # conv5 = self.dconv_down5(x)
        # x = self.maxpool(conv5)

        # conv6 = self.dconv_down6(x)
        # x = self.maxpool(conv6)

        # x = self.dconv_down7(x)

        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up6(x)
        x = self.upsample(x)
        x = torch.cat([x, t_layer2], dim=1)

        x = self.dconv_up5(x)
        x = self.upsample(x)
        x = torch.cat([x, t_layer1], dim=1)

        # x = self.dconv_up4(x)
        # x = self.upsample(x)
        # x = torch.cat([x, conv3], dim=1)

        # x = self.dconv_up3(x)
        # x = self.upsample(x)
        # x = torch.cat([x, conv2], dim=1)

        # x = self.dconv_up2(x)
        # x = self.upsample(x)
        # x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        x = self.conv_last(x)

        return x
