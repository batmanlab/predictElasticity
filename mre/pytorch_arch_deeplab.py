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
        print('new block')
        print(inp.size())
        print(self.rep)
        x = self.rep(inp)
        print(x.size())

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        print(skip.size())
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
        self.conv1 = nn.Conv3d(in_channels, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(64)

        self.block1 = Block(64, 128, reps=2, stride=2, start_with_relu=False, shrink_z=False)
        self.block2 = Block(128, 256, reps=2, stride=2, start_with_relu=False,
                            grow_first=True, shrink_z=False)
        # self.block2 = Block(128, 128, reps=2, stride=entry_block3_stride,
        #                     start_with_relu=True,
        #                     grow_first=True, is_last=True)
        self.block3 = Block(256, 512, reps=2, stride=entry_block3_stride,
                            start_with_relu=True, grow_first=True, is_last=True, shrink_z=False)

        # Middle flow
        self.block4  = Block(512, 512, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block5  = Block(512, 512, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block6  = Block(512, 512, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block7  = Block(512, 512, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block8  = Block(512, 512, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block9  = Block(512, 512, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block10 = Block(512, 512, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block11 = Block(512, 512, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block12 = Block(512, 512, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block13 = Block(512, 512, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block14 = Block(512, 512, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block15 = Block(512, 512, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block16 = Block(512, 512, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block17 = Block(512, 512, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block18 = Block(512, 512, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)
        self.block19 = Block(512, 512, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True)

        # Exit flow
        self.block20 = Block(512, 1024, reps=2, stride=1, dilation=exit_block_dilations[0],
                             start_with_relu=True, grow_first=False, is_last=True)

        self.conv3 = SeparableConv3d(1024, 1536, 3, stride=1, dilation=exit_block_dilations[1])
        self.bn3 = nn.BatchNorm3d(1536)

        self.conv4 = SeparableConv3d(1536, 1536, 3, stride=1, dilation=exit_block_dilations[1])
        self.bn4 = nn.BatchNorm3d(1536)

        self.conv5 = SeparableConv3d(1536, 2048, 3, stride=1, dilation=exit_block_dilations[1])
        self.bn5 = nn.BatchNorm3d(2048)

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


if __name__ == "__main__":
    model = AlignedXception(output_stride=16)
    input = torch.rand(1, 9, 32, 256, 256)
    output, low_level_feat = model(input)
    print(output.size())
    print(low_level_feat.size())
