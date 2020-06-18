# Attempt at making DeepLabv3+ 3D
# https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/xception.py
# https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/modeling/backbone/xception.py
# Ordinal:
# https://github.com/dontLoveBugs/DORN_pytorch/blob/master/network/DORN_nyu.py
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
                 bias=True, norm='bn'):
        super(SeparableConv3d, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding, dilation,
                               groups=in_channels, bias=bias)
        self.pointwise = nn.Conv3d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        if norm == 'bn':
            self.norm = nn.BatchNorm3d(out_channels)
        elif norm == 'gn':
            groups = int(min(out_channels/2, 32))
            self.norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)
        else:
            raise ValueError(f'Incorrect norm "{norm}" specified')

    def forward(self, x):
        # x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.pointwise(x)
        x = self.norm(x)
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
                 start_with_relu=True, grow_first=True, is_last=False, shrink_z=True, norm='bn'):
        super(Block, self).__init__()

        stride_3d = stride
        if out_channels != in_channels or stride != 1:
            if stride != 1:
                if shrink_z:
                    stride_3d = 2
                else:
                    stride_3d = (1, 2, 2)
            self.skip = nn.Conv3d(in_channels, out_channels, 1, stride=stride_3d, bias=False)
            if norm == 'bn':
                self.skipnorm = nn.BatchNorm3d(out_channels)
            elif norm == 'gn':
                groups = int(min(out_channels/2, 32))
                self.skipnorm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)
            else:
                raise ValueError(f'Incorrect norm "{norm}" specified')
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_channels
        padding = dilation
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv3d(in_channels, out_channels, 3, 1, padding, dilation=dilation,
                                       norm=norm))
            filters = out_channels

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv3d(filters, filters, 3, 1, padding, dilation=dilation,
                                       norm=norm))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv3d(in_channels, out_channels, 3, 1, padding, dilation=dilation,
                                       norm=norm))

        # if stride == 2:
        #     rep.append(self.relu)
        #     rep.append(SeparableConv3d(out_channels, out_channels, 4, 2))
        #     rep.append(nn.GroupNorm(out_channels))

        if stride != 1:
            if shrink_z:
                stride_3d = 2
            else:
                stride_3d = (1, 2, 2)
            rep.append(self.relu)
            rep.append(SeparableConv3d(out_channels, out_channels, 3, stride_3d, 1, norm=norm))

        if stride == 1 and is_last:
            rep.append(self.relu)
            rep.append(SeparableConv3d(out_channels, out_channels, 3, 1, 1, norm=norm))

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
            skip = self.skipnorm(skip)
        else:
            skip = inp

        # print(skip.size())
        x = x + skip

        return x


class AlignedXception(nn.Module):
    """
    Modified Alighed Xception
    """
    def __init__(self, in_channels, output_stride, norm='bn'):
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
        if norm == 'bn':
            self.norm1 = nn.BatchNorm3d(18)
        elif norm == 'gn':
            self.norm1 = nn.GroupNorm(18, 18)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(18, 32, 3, stride=1, padding=1, bias=False)
        if norm == 'bn':
            self.norm2 = nn.BatchNorm3d(32)
        elif norm == 'gn':
            self.norm2 = nn.GroupNorm(32, 32)

        self.block1 = Block(32, 64, reps=2, stride=2, start_with_relu=False, shrink_z=False,
                            norm=norm)
        self.block2 = Block(64, 128, reps=2, stride=2, start_with_relu=False,
                            grow_first=True, shrink_z=False, norm=norm)
        # self.block2 = Block(128, 128, reps=2, stride=entry_block3_stride,
        #                     start_with_relu=True,
        #                     grow_first=True, is_last=True)
        self.block3 = Block(128, 256, reps=2, stride=entry_block3_stride,
                            start_with_relu=True, grow_first=True, is_last=True, shrink_z=False,
                            norm=norm)

        # Middle flow
        self.block4  = Block(256, 256, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True, norm=norm)
        self.block5  = Block(256, 256, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True, norm=norm)
        self.block6  = Block(256, 256, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True, norm=norm)
        self.block7  = Block(256, 256, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True, norm=norm)
        self.block8  = Block(256, 256, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True, norm=norm)
        self.block9  = Block(256, 256, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True, norm=norm)
        self.block10 = Block(256, 256, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True, norm=norm)
        self.block11 = Block(256, 256, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True, norm=norm)
        self.block12 = Block(256, 256, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True, norm=norm)
        self.block13 = Block(256, 256, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True, norm=norm)
        self.block14 = Block(256, 256, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True, norm=norm)
        self.block15 = Block(256, 256, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True, norm=norm)
        self.block16 = Block(256, 256, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True, norm=norm)
        self.block17 = Block(256, 256, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True, norm=norm)
        self.block18 = Block(256, 256, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True, norm=norm)
        self.block19 = Block(256, 256, reps=3, stride=1, dilation=middle_block_dilation,
                             start_with_relu=True, grow_first=True, norm=norm)

        # Exit flow
        self.block20 = Block(256, 512, reps=2, stride=1, dilation=exit_block_dilations[0],
                             start_with_relu=True, grow_first=False, is_last=True, norm=norm)

        self.conv3 = SeparableConv3d(512, 768, 3, stride=1, dilation=exit_block_dilations[0],
                                     padding=1, norm=norm)

        self.conv4 = SeparableConv3d(768, 1024, 3, stride=1, dilation=exit_block_dilations[0],
                                     padding=1, norm=norm)

        self.conv5 = SeparableConv3d(1024, 1280, 3, stride=1, dilation=exit_block_dilations[0],
                                     padding=1, norm=norm)

        # Init weights
        self._init_weight()

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
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
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.relu(x)

        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class _ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation, norm='bn'):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False,
                                     groups=in_channels)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                   dilation=1, bias=False)

        if norm == 'bn':
            self.norm = nn.BatchNorm3d(out_channels)
        elif norm == 'gn':
            groups = int(min(out_channels/2, 32))
            self.norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.pointwise(x)
        x = self.norm(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, output_stride, norm='bn'):
        super(ASPP, self).__init__()
        in_channels = 1280
        if output_stride == 16:
            # dilations = [1, 6, 12, 18]
            dilations = [1, 2, 4, 6]
        elif output_stride == 8:
            # dilations = [1, 12, 24, 36]
            dilations = [1, 2, 4, 6]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(in_channels, 256, 1, padding=0, dilation=dilations[0], norm=norm)
        self.aspp2 = _ASPPModule(in_channels, 256, 3, padding=dilations[1], dilation=dilations[1],
                                 norm=norm)
        self.aspp3 = _ASPPModule(in_channels, 256, 3, padding=dilations[2], dilation=dilations[2],
                                 norm=norm)
        self.aspp4 = _ASPPModule(in_channels, 256, 3, padding=dilations[3], dilation=dilations[3],
                                 norm=norm)

        if norm == 'bn':
            self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)),
                                                 nn.Conv3d(in_channels, 256, 1, stride=1,
                                                           bias=False),
                                                 nn.BatchNorm3d(256),
                                                 nn.ReLU())
        elif norm == 'gn':
            self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)),
                                                 nn.Conv3d(in_channels, 256, 1, stride=1,
                                                           bias=False),
                                                 nn.GroupNorm(64, 256),
                                                 nn.ReLU())
        self.conv1 = nn.Conv3d(1280, 256, 1, bias=False)
        if norm == 'bn':
            self.norm = nn.BatchNorm3d(256)
        elif norm == 'gn':
            self.norm = nn.GroupNorm(64, 256)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.5)
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
        x = self.norm(x)
        x = self.relu(x)

        # return self.dropout(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Decoder(nn.Module):
    def __init__(self, out_channels, norm='bn', do_clinical=False):
        super(Decoder, self).__init__()
        low_level_inplanes = 64

        self.conv1 = nn.Conv3d(low_level_inplanes, 32, 1, bias=False)
        self.relu = nn.ReLU()
        self.do_clinical = do_clinical
        if norm == 'bn':
            self.norm = nn.BatchNorm3d(32)
            if self.do_clinical:
                self.last_conv = nn.Sequential(nn.Conv3d(256+32+14, 128, kernel_size=3, stride=1,
                                                         padding=1, bias=False),
                                               nn.BatchNorm3d(128),
                                               nn.ReLU(),
                                               nn.Conv3d(128, 128, kernel_size=3, stride=1,
                                                         padding=1, bias=False),
                                               nn.BatchNorm3d(128),
                                               nn.ReLU(),
                                               nn.Conv3d(128, out_channels, kernel_size=1,
                                                         stride=1)
                                               )
            else:
                self.last_conv = nn.Sequential(nn.Conv3d(256+32, 128, kernel_size=3, stride=1,
                                                         padding=1, bias=False),
                                               nn.BatchNorm3d(128),
                                               nn.ReLU(),
                                               nn.Conv3d(128, 128, kernel_size=3, stride=1,
                                                         padding=1, bias=False),
                                               nn.BatchNorm3d(128),
                                               nn.ReLU(),
                                               nn.Conv3d(128, out_channels, kernel_size=1,
                                                         stride=1)
                                               )
        elif norm == 'gn':
            self.norm = nn.GroupNorm(32, 32)
            if self.do_clinical:
                self.last_conv = nn.Sequential(nn.Conv3d(256+32+14, 128, kernel_size=3, stride=1,
                                                         padding=1, bias=False),
                                               nn.GroupNorm(64, 128),
                                               nn.ReLU(),
                                               nn.Conv3d(128, 128, kernel_size=3, stride=1,
                                                         padding=1, bias=False),
                                               nn.GroupNorm(64, 128),
                                               nn.ReLU(),
                                               nn.Conv3d(128, out_channels, kernel_size=1,
                                                         stride=1)
                                               )
            else:
                self.last_conv = nn.Sequential(nn.Conv3d(256+32, 128, kernel_size=3, stride=1,
                                                         padding=1, bias=False),
                                               nn.GroupNorm(64, 128),
                                               nn.ReLU(),
                                               nn.Conv3d(128, 128, kernel_size=3, stride=1,
                                                         padding=1, bias=False),
                                               nn.GroupNorm(64, 128),
                                               nn.ReLU(),
                                               nn.Conv3d(128, out_channels,
                                                         kernel_size=1, stride=1))
        self._init_weight()

    def forward(self, x, low_level_feat, clinical=None):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.norm(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='trilinear', align_corners=True)
        if self.do_clinical:
            x = torch.cat((x, low_level_feat, clinical), dim=1)
            x = self.last_conv(x)
        else:
            x = torch.cat((x, low_level_feat), dim=1)
            x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DecoderFeatures(nn.Module):
    def __init__(self, out_channels, norm='bn', do_clinical=False):
        super(DecoderFeatures, self).__init__()
        low_level_inplanes = 64

        self.conv1 = nn.Conv3d(low_level_inplanes, 32, 1, bias=False)
        self.relu = nn.ReLU()
        self.do_clinical = do_clinical
        self.norm = nn.BatchNorm3d(32)
        self.last_conv = nn.Sequential(nn.Conv3d(256+32+14, 128, kernel_size=3, stride=1,
                                                 padding=1, bias=False),
                                       nn.BatchNorm3d(128),
                                       nn.ReLU(),
                                       nn.Conv3d(128, 128, kernel_size=3, stride=1,
                                                 padding=1, bias=False),
                                       )
        self._init_weight()

    def forward(self, x, low_level_feat, clinical=None):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.norm(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='trilinear', align_corners=True)
        x = torch.cat((x, low_level_feat, clinical), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class OrdinalRegressionLayerSigmoid(nn.Module):
    def __init__(self):
        super(OrdinalRegressionLayerSigmoid, self).__init__()

    def forward(self, x):
        """
        :param x: N X D X H X W X C, N is batch_size, C is channels of features
        :return: ord_labels is ordinal outputs for each spatial locations,
                 size is N x H X W X K (K is interval of SID)
                 decode_label is the ordinal labels for each position of Image I
        """
        # N, K, D, H, W = x.size()

        """
        replace iter with matrix operation
        fast speed methods
        """
        # x = torch.clamp(x, min=-1e8, max=1e8)  # prevent nans

        ord_c = torch.sigmoid(x)
        # print(ord_c.requires_grad)

        decode_c = torch.sum((ord_c > 0.5), dim=1).unsqueeze(1)
        # print(decode_c.requires_grad)
        # decode_c = torch.sum(ord_c1, dim=1).view(-1, 1, H, W)
        return decode_c, ord_c.float()


# https://github.com/dontLoveBugs/DORN_pytorch/blob/master/network/DORN_nyu.py
class OrdinalRegressionLayer(nn.Module):
    def __init__(self):
        super(OrdinalRegressionLayer, self).__init__()

    def forward(self, x):
        """
        :param x: N X D X H X W X C, N is batch_size, C is channels of features
        :return: ord_labels is ordinal outputs for each spatial locations,
                 size is N x H X W X C (C = 2K, K is interval of SID)
                 decode_label is the ordinal labels for each position of Image I
        """
        N, C, D, H, W = x.size()
        ord_num = C // 2

        """
        replace iter with matrix operation
        fast speed methods
        """
        A = x[:, ::2, :, :, :].clone()
        B = x[:, 1::2, :, :, :].clone()

        A = A.view(N, 1, ord_num * D * H * W)
        B = B.view(N, 1, ord_num * D * H * W)

        C = torch.cat((A, B), dim=1)
        C = torch.clamp(C, min=1e-8, max=1e8)  # prevent nans

        ord_c = nn.functional.softmax(C, dim=1)
        # print(ord_c.requires_grad)

        ord_c1 = ord_c[:, 1, :].clone()
        ord_c1 = ord_c1.view(-1, ord_num, D, H, W)
        # print(ord_c1.requires_grad)
        # print('ord > 0.5 size:', (ord_c1 > 0.5).size())
        decode_c = torch.sum((ord_c1 > 0.5), dim=1).view(-1, 1, D, H, W)
        # print(decode_c.requires_grad)
        # decode_c = torch.sum(ord_c1, dim=1).view(-1, 1, H, W)
        return decode_c, ord_c1.float()


class DeepLab(nn.Module):
    def __init__(self, in_channels, out_channels, output_stride=8, do_ord=False, norm='bn',
                 do_clinical=False):
        super(DeepLab, self).__init__()

        self.backbone = AlignedXception(in_channels, output_stride, norm=norm)
        self.aspp = ASPP(output_stride, norm=norm)
        self.decoder = Decoder(out_channels, norm=norm, do_clinical=do_clinical)
        self.do_ord = do_ord
        if self.do_ord:
            self.ord_layer = OrdinalRegressionLayerSigmoid()
        self.do_clinical = do_clinical

    def forward(self, input, clinical=None):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat, clinical)

        if self.do_ord:
            depth_labels, ord_labels = self.ord_layer(x)
            depth_labels = F.interpolate(depth_labels.float(), size=input.size()[2:],
                                         mode='trilinear', align_corners=True)
            ord_labels = F.interpolate(ord_labels, size=input.size()[2:],
                                       mode='trilinear', align_corners=True)

            return depth_labels, ord_labels
        else:
            x = F.interpolate(x, size=input.size()[2:], mode='trilinear', align_corners=True)

            return x


class DeepLabFeatures(nn.Module):
    def __init__(self, in_channels, out_channels, output_stride=8, do_ord=False, norm='bn',
                 do_clinical=False):
        super(DeepLabFeatures, self).__init__()

        self.backbone = AlignedXception(in_channels, output_stride, norm=norm)
        self.aspp = ASPP(output_stride, norm=norm)
        self.decoder = DecoderFeatures(out_channels, norm=norm, do_clinical=do_clinical)
        self.do_ord = do_ord
        self.do_clinical = do_clinical

    def forward(self, input, clinical=None):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat, clinical)

        return x


if __name__ == "__main__":
    model = AlignedXception(output_stride=16)
    input = torch.rand(1, 9, 32, 256, 256)
    output, low_level_feat = model(input)
    print(output.size())
    print(low_level_feat.size())
