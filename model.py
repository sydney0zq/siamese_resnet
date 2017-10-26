#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017-10-25 20:42 zq <zq@mclab>
#
# Distributed under terms of the MIT license.

"""
Simaese network in pytorch.
"""

import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms

class SiameseBranchNetwork(nn.Module):
    def __init__(self):
        super(SiameseBranchNetwork, self).__init__()
        # 7x7 and 512 channels
        self.resnet18 = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])
        self.branch1 = nn.Sequential(
                            nn.Conv2d(512, 20, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(20, 10, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(10, 5, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(
                            nn.Conv2d(512, 20, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(20, 10, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(10, 5, kernel_size=3, stride=1, padding=1))

    def forward(self, input_a, input_b):
        output_a = self.resnet18(input_a)
        output_a = self.branch1(output_a)
        output_b = self.resnet18(input_b)
        output_b = self.branch2(output_b)
        return output_a, output_b
