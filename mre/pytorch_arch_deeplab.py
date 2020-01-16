# Attempt at making DeepLabv3+ 3D
# https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/xception.py
# https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/modeling/backbone/xception.py
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from tensorboardX import SummaryWriter
from mre.CoordConv import CoordConv


class SeparableConv3d(nn.Module):
    '''Standard separable 3d conv block.  A depthwise 3D conv followed by a 1x1 conv to mix the
    channels.  If dilation is specified, this becomes Atrous Separable Convolution, as per
    DeepLabv3+.'''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1,
                 bias=True):
        super(SeparableConv3d, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding, dilation,
                               groups=in_channels, bias=bias)
        self.bn = nn.BatchNorm3d(in_channels)
        self.pointwise = nn.Conv3d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        # x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


def fixed_padding(inputs, kernel_size, dilation):
    # unclear why this would be needed
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, reps, stride=1, dilation=1,
                 start_with_relu=True, grow_first=True, is_last=False, shrink_z=True):
        super(Block, self).__init__()

        stride_3d = stride
        if out_channels != in_channels or stride != 1:
            if stride != 1:
                if shrink_z:
                    stride_3d = 2
                else:
                    stride_3d = (1, 2, 2)
            self.skip = nn.Conv3d(in_channels, out_channels, 1, stride=stride_3d, bias=False)
            self.skipbn = nn.BatchNorm3d(out_channels)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_channels
        padding = dilation
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv3d(in_channels, out_channels, 3, 1, padding, dilation=dilation))
            rep.append(nn.BatchNorm3d(out_channels))
            filters = out_channels

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv3d(filters, filters, 3, 1, padding, dilation=dilation))
            rep.append(nn.BatchNorm3d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv3d(in_channels, out_channels, 3, 1, padding, dilation=dilation))
            rep.append(nn.BatchNorm3d(out_channels))

        # if stride == 2:
        #     rep.append(self.relu)
        #     rep.append(SeparableConv3d(out_channels, out_channels, 4, 2))
        #     rep.append(nn.BatchNorm3d(out_channels))

        if stride != 1:
            if shrink_z:
                stride_3d = 2
            else:
                stride_3d = (1, 2, 2)
            rep.append(self.relu)
            rep.append(SeparableConv3d(out_channels, out_channels, 3, stride_3d, 1))
            rep.append(nn.BatchNorm3d(out_channels))

        if stride == 1 and is_last:
            rep.append(self.relu)
            rep.append(SeparableConv3d(out_channels, out_channels, 3, 1, 1))
            rep.append(nn.BatchNorm3d(out_channels))

        if not start_with_relu:
            rep = rep[1:]

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        # print('new block')
        # print(inp.size())
        # print(self.rep)
        x = self.rep(inp)
        # print(x.size())

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        # print(skip.size())
        x = x + skip

        return x


class AlignedXception(nn.Module):
    """
    Modified Alighed Xception
    """
    def __init__(self, in_channels, output_stride):
        super(AlignedXception, self).__init__()

        if output_stride == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 2)
        elif output_stride == 8:
            # entry_block3_stride = 1
            entry_block3_stride = 2
            middle_block_dilation = 2
            exit_block_dilations = (1, 2)
        else:
            raise NotImplementedError

        # Entry flow
        self.conv1 = nn.Conv3d(in_channels, 18, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(18)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(18, 32, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(32)

        self.block1 = Block(32, 64, reps=2, stride=2, start_with_relu=False, shrink_z=False)
        self.block2 = Block(64, 128, reps=2, stride=2, start_with_relu=False,
                            grow_first=True, shrink_z=False)
        # self.block2 = Block(128, 128, reps=2, stride=entry_block3_stride,
        #                     start_with_relu=True,
        #                     grow_first=True, is_last=True)
        self.block3 = Block(128, 256, reps=2, stride=entry_block3_stride,
                            start_with_relu=True, grow_first=True, is_last=True, shrink_z=False)

        # Middle flow
        self.block4  = Block(256, 256, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block5  = Block(256, 256, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block6  = Block(256, 256, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block7  = Block(256, 256, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block8  = Block(256, 256, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block9  = Block(256, 256, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block10 = Block(256, 256, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block11 = Block(256, 256, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block12 = Block(256, 256, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block13 = Block(256, 256, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block14 = Block(256, 256, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block15 = Block(256, 256, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block16 = Block(256, 256, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block17 = Block(256, 256, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block18 = Block(256, 256, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block19 = Block(256, 256, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)

        # Exit flow
        self.block20 = Block(256, 512, reps=2, stride=1, dilation=exit_block_dilations[0],
                             start_with_relu=True, grow_first=False, is_last=True)

        self.conv3 = SeparableConv3d(512, 512, 3, stride=1, dilation=exit_block_dilations[0],
                                     padding=1)
        self.bn3 = nn.BatchNorm3d(512)

        self.conv4 = SeparableConv3d(512, 512, 3, stride=1, dilation=exit_block_dilations[0],
                                     padding=1)
        self.bn4 = nn.BatchNorm3d(512)

        self.conv5 = SeparableConv3d(512, 512, 3, stride=1, dilation=exit_block_dilations[0],
                                     padding=1)
        self.bn5 = nn.BatchNorm3d(512)

        # Init weights
        self._init_weight()

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        # add relu here
        x = self.relu(x)
        low_level_feat = x
        x = self.block2(x)
        x = self.block3(x)

        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)

        # Exit flow
        x = self.block20(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class _ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False,
                                     groups=in_channels)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                   dilation=1, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.pointwise(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, output_stride):
        super(ASPP, self).__init__()
        in_channels = 512
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(in_channels, 256, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(in_channels, 256, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(in_channels, 256, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(in_channels, 256, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)),
                                             nn.Conv3d(in_channels, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm3d(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv3d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='trilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Decoder(nn.Module):
    def __init__(self, out_channels):
        super(Decoder, self).__init__()
        low_level_inplanes = 64

        self.conv1 = nn.Conv3d(low_level_inplanes, 32, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv3d(256+32, 128, kernel_size=3, stride=1,
                                                 padding=1, bias=False),
                                       nn.BatchNorm3d(128),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv3d(128, 128, kernel_size=3, stride=1,
                                                 padding=1, bias=False),
                                       nn.BatchNorm3d(128),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv3d(128, out_channels, kernel_size=1, stride=1))
        self._init_weight()

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='trilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLab(nn.Module):
    def __init__(self, in_channels, out_channels, output_stride=8):
        super(DeepLab, self).__init__()

        self.backbone = AlignedXception(in_channels, output_stride)
        self.aspp = ASPP(output_stride)
        self.decoder = Decoder(out_channels)

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='trilinear', align_corners=True)

        return x


if __name__ == "__main__":
    model = AlignedXception(output_stride=16)
    input = torch.rand(1, 9, 32, 256, 256)
    output, low_level_feat = model(input)
    print(output.size())
    print(low_level_feat.size())
