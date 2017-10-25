#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017-10-25 21:10 zq <zq@mclab>
#
# Distributed under terms of the MIT license.

"""
Dataset wrapper for siamese network.
"""

class Dataset(object):
    def __init__(self, im_a, im_b, label):
        self.size = label.shape[0]
        self.im_a = torch.from_numpy(im_a)
        self.im_b = torch.from_numpy(im_b)
        self.label = self.from_numpy(label)
    def __getitem__(self, index):
        return (self.im_a[index], self.im_b[index], self.label[index])
    def __len__(self):
        return self.size

























