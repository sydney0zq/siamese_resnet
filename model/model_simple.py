#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017-10-25 20:42 zq <zq@mclab>
#
# Distributed under terms of the MIT license.

"""
Simaese bitmap network in pytorch.
"""

import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms

class SiameseBranchNetwork(nn.Module):
    def __init__(self):
        super(SiameseBranchNetwork, self).__init__()
        self.feature = nn.Sequential(
                            nn.Conv2d(3, 50, kernel_size=3, stride=2, padding=1), # to 256x256
                            nn.ReLU(inplace=True),
                            nn.Conv2d(50, 40, kernel_size=3, stride=2, padding=1), # to 128x128
                            nn.ReLU(inplace=True),
                            nn.Conv2d(40, 40, kernel_size=3, stride=2, padding=1), # to 64x64
                            nn.ReLU(inplace=True),
                            nn.Conv2d(40, 30, kernel_size=3, stride=2, padding=1), # to 32x32
                            nn.ReLU(inplace=True),
                            nn.Conv2d(30, 20, kernel_size=3, stride=2, padding=1), # to 16x16
                            nn.ReLU(inplace=True))
        self.branch1 = nn.Sequential(
                            nn.Conv2d(20, 10, kernel_size=3, stride=2, padding=0), # to 7x7
                            nn.ReLU(inplace=True),
                            nn.Conv2d(10, 5, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(
                            nn.Conv2d(20, 10, kernel_size=3, stride=2, padding=0), 
                            nn.ReLU(inplace=True),
                            nn.Conv2d(10, 5, kernel_size=3, stride=1, padding=1))

    def forward(self, input_a, input_b):
        output_a = self.feature(input_a)
        output_a = self.branch1(output_a)
        output_b = self.feature(input_b)
        output_b = self.branch2(output_b)
        return output_a, output_b
