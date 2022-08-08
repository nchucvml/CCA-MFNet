import torch.nn as nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.autograd import Variable
affine_par = True
import functools

import sys, os

import matplotlib # 20210506
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

#from cc_attention import CrissCrossAttention 
from .CC import CC_module as CrissCrossAttention
from utils.pyt_utils import load_model

from inplace_abn import InPlaceABN, InPlaceABNSync
from Synchronized.sync_batchnorm import SynchronizedBatchNorm2d as SyncBN
BatchNorm2d = SyncBN#functools.partial(InPlaceABNSync, activation='identity')
from einops import rearrange

def outS(i):
    i = int(i)
    i = (i+1)/2
    i = int(np.ceil((i+1)/2.0))
    i = (i+1)/2
    return i

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual      
        out = self.relu_inplace(out)

        return out

class RCCAModule2(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(RCCAModule2, self).__init__()
        inter_channels = in_channels // 4
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   BatchNorm2d(inter_channels),nn.ReLU(inplace=False))
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   BatchNorm2d(inter_channels),nn.ReLU(inplace=False))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            BatchNorm2d(out_channels),nn.ReLU(inplace=False),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

        print(self.bottleneck)

    def forward(self, x, recurrence=1):
        output = self.conva(x)
        for i in range(recurrence):
            output = self.cca(output)
        output = self.convb(output)

        output = torch.cat([x, output], 1)
        return output


class CCA_MFNet(nn.Module):
    def __init__(self, block, layers, num_classes, criterion, recurrence):
        self.inplanes = 128
        super(CCA_MFNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, multi_grid=(1,1,1))

        self.conv4 = nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU(inplace=False)
        self.conv5 = nn.Conv2d(2560, 512, kernel_size=3, stride=1, padding=2, dilation=2)
        self.relu5 = nn.ReLU(inplace=False)
        self.conv6 = nn.Conv2d(2560, 512, kernel_size=3, stride=1, padding=4, dilation=4)
        self.relu6 = nn.ReLU(inplace=False)
        self.conv7 = nn.Conv2d(2560, 512, kernel_size=3, stride=1, padding=8, dilation=8)
        self.bn4 = BatchNorm2d(2048)
        self.relu7 = nn.ReLU(inplace=False)
        self.DropOut2D1 = nn.Dropout2d(0.1)

        self.head = RCCAModule2(2048, 512, num_classes)

        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(512),nn.ReLU(inplace=False),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

        self.criterion = criterion
        self.recurrence = recurrence

        self.upsamp0 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.GAvgPool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.GAvgPool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1x1_1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv8 = nn.Conv2d(2560, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = BatchNorm2d(256)
        self.relu8 = nn.ReLU(inplace=False)
        self.upsamp1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv9 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = BatchNorm2d(256)
        self.relu9 = nn.ReLU(inplace=False)
        self.upsamp2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dsn2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(256), nn.ReLU(inplace=False),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion,affine = affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index%len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x, labels=None):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        a_out = x
        x = self.layer2(x)
        b_out = x
        x = self.layer3(x)
        x_dsn = self.dsn(x)
        x = self.layer4(x)

        d1 = self.conv4(x)
        y = torch.cat([x, d1], 1)
        y = self.relu4(y)
        d2 = self.conv5(y)
        y = torch.cat([x, d2], 1)
        y = self.relu5(y)
        d4 = self.conv6(y)
        y = torch.cat([x, d4], 1)
        y = self.relu6(y)
        d8 = self.conv7(y)
        x = torch.cat([d1, d2, d4, d8], 1)
        x = self.bn4(x)
        x = self.relu7(x)
        x = self.DropOut2D1(x)

        x = self.head(x, self.recurrence)
        x_dsn = self.upsamp0(x_dsn)

        a_out = self.GAvgPool1(a_out)
        b_out = self.conv1x1_1(b_out)
        b_out = self.GAvgPool2(b_out)
        x = self.conv8(x)
        x = self.bn5(x)
        x = self.relu8(x)
        x1 = torch.mul(x, b_out)
        x = torch.add(x, x1)
        x = self.upsamp1(x)
        x = self.conv9(x)
        x = self.bn6(x)
        x = self.relu9(x)
        x2 = torch.mul(x, a_out)
        x = torch.add(x, x2)
        x = self.upsamp2(x)

        x_dsn2 = self.dsn2(x)
        outs = [x_dsn2, x_dsn]

        if self.criterion is not None and labels is not None:
            # print("train.py")
            return self.criterion(outs, labels)
        else:
            # print("evaluate.py")
            return outs

def Seg_Model(num_classes, criterion=None, pretrained_model=None, recurrence=0, **kwargs):
    model = CCA_MFNet(Bottleneck, [3, 4, 6, 3], num_classes, criterion, recurrence)  # ResNet50
    # model = CCA_MFNet(Bottleneck, [3, 4, 23, 3], num_classes, criterion, recurrence) # ResNet101

    if pretrained_model is not None:
        model = load_model(model, pretrained_model)


    return model
