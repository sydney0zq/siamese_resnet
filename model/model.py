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
        super(SiameseBranchNetwork, self).__init__()
        # 7x7 and 512 channels
        self.resnet18 = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])
        # 512x14x14 feature map
        self.regression = nn.Sequential(
                            nn.Conv2d(512, 20, kernel_size=4, stride=2, padding=0), # to 7x7
                            nn.ReLU(inplace=True),
                            nn.Conv2d(20, 10, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(10, 5, kernel_size=3, stride=1, padding=1))

    def forward(self, input_a, input_b):
        output_a = self.resnet18(input_a)
        output_b = self.resnet18(input_b)
        concated_fea = torch.cat([outputa, outputb], dim=2)
        ouput = self.regression(concated_fea)
        return output
