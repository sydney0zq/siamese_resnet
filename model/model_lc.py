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
                            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=0), # 7x7
                            nn.ReLU(inplace=True),
                            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(32, 7, kernel_size=3, stride=1, padding=1))

    def forward(self, inputa, inputb):
        outputa = self.resnet18(inputa)
        outputb = self.resnet18(inputb)
        multi_fea = output_a * output_b
        #concated_fea = torch.cat([outputa, outputb], dim=1) # [batch_size, 1024, 16, 16]
        output = self.regression(concated_fea)
        return output
