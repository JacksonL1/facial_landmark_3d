#!/usr/bin/env python3
# coding: utf-8

from __future__ import division

""" 
Creates a MobileNet Model as defined in:
Andrew G. Howard Menglong Zhu Bo Chen, et.al. (2017). 
MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. 
Copyright (c) Yang Lu, 2017

Modified By cleardusk
"""
import math
import torch
import torch.nn as nn
import cv2
from utils.inference import crop_img


class DepthWiseBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, prelu=False):
        super(DepthWiseBlock, self).__init__()
        inplanes, planes = int(inplanes), int(planes)
        self.conv_dw = nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1, stride=stride, groups=inplanes,
                                 bias=False)
        self.bn_dw = nn.BatchNorm2d(inplanes)
        self.conv_sep = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_sep = nn.BatchNorm2d(planes)
        if prelu:
            self.relu = nn.PReLU()
        else:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv_dw(x)
        out = self.bn_dw(out)
        out = self.relu(out)

        out = self.conv_sep(out)
        out = self.bn_sep(out)
        out = self.relu(out)

        return out


class MobileNet(nn.Module):
    def __init__(self, widen_factor=1.0, num_classes=1000, prelu=False, input_channel=3):
        """ Constructor
        Args:
            widen_factor: config of widen_factor
            num_classes: number of classes
        """
        super(MobileNet, self).__init__()
        block = DepthWiseBlock
        self.conv1 = nn.Conv2d(input_channel, int(32 * widen_factor), kernel_size=3, stride=2, padding=1,
                               bias=False)

        self.bn1 = nn.BatchNorm2d(int(32 * widen_factor))
        if prelu:
            self.relu = nn.PReLU()
        else:
            self.relu = nn.ReLU(inplace=True)

        self.dw2_1 = block(32 * widen_factor, 64 * widen_factor, prelu=prelu)
        self.dw2_2 = block(64 * widen_factor, 128 * widen_factor, stride=2, prelu=prelu)

        self.dw3_1 = block(128 * widen_factor, 128 * widen_factor, prelu=prelu)
        self.dw3_2 = block(128 * widen_factor, 256 * widen_factor, stride=2, prelu=prelu)

        self.dw4_1 = block(256 * widen_factor, 256 * widen_factor, prelu=prelu)
        self.dw4_2 = block(256 * widen_factor, 512 * widen_factor, stride=2, prelu=prelu)

        self.dw5_1 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_2 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_3 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_4 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_5 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_6 = block(512 * widen_factor, 1024 * widen_factor, stride=2, prelu=prelu)

        self.dw6 = block(1024 * widen_factor, 1024 * widen_factor, prelu=prelu)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(int(1024 * widen_factor), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.dw2_1(x)
        x = self.dw2_2(x)
        x = self.dw3_1(x)
        x = self.dw3_2(x)
        x = self.dw4_1(x)
        x = self.dw4_2(x)
        x = self.dw5_1(x)
        x = self.dw5_2(x)
        x = self.dw5_3(x)
        x = self.dw5_4(x)
        x = self.dw5_5(x)
        x = self.dw5_6(x)
        x = self.dw6(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def mobilenet(checkpoint_fp, num_classes=240, input_channel=3):
    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model = MobileNet(widen_factor=1.0, num_classes=num_classes,  input_channel=input_channel)
    model_dict = model.state_dict()
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model.load_state_dict(model_dict)
    model = model.cuda()
    model.eval()
    return model
