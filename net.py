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

class SiameseNetwork(nn.Module):
    def __init__(SiameseNetwork, self).__init__():
        # 7x7 and 512 channels
        self.resnet18 = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])
        self.branch = nn.Sequential(
                            nn.Conv2d(512, 20, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(20, 10, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(inplace.True),
                            nn.Conv2d(10, 5, kernel_size=3, stride=1, padding=1))

    def forward_once(self, x):
        output = self.resnet18(x)
        output = self.branch(output)
        return output
    
    def forward(self, input_a, input_b):
        output_a = self.forward_once(input_a)
        output_b = self.forward_once(input_b)
        return output_a, output_b
