#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017-10-25 20:42 zq <zq@mclab>
#
# Distributed under terms of the MIT license.

"""
Find difference of two similar image through learning in pytorch.
"""

import torch.nn as nn
import torchvision
import torch
from torchvision import datasets, models, transforms

class DiffNetwork(nn.Module):
    def __init__(self):
        super(DiffNetwork, self).__init__()
        # 16x16 and 512 channels
        self.resnet18 = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])
        self.regression = nn.Sequential(
                            # To 14x14
                            nn.Conv2d(1024, 512, kernel_size=4, stride=1, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(512, 512, kernel_size=4, stride=1, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(512, 7, kernel_size=3, stride=1, padding=1)) # 7x14x14

    def forward(self, inputa, inputb):
        outputa = self.resnet18(inputa)
        outputb = self.resnet18(inputb)
        dualsub = torch.cat([outputa-outputb, outputb-outputa], dim=1)
        output = self.regression(dualsub)
        return output
