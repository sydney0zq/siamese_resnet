#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017-10-26 22:09 zq <zq@mclab>
#
# Distributed under terms of the MIT license.

"""
Customed part of YOLO loss.
"""

import torch
import numpy as np
import time
import matplotlib.pyplot as plt

from torch.autograd import Variable


__all__ = ['YOLO_loss', 'criterion']

"""
How to get value from GPU RAM
    You first need to get tensor out of the variable using .data 
    Then you need to move the tensor to cpu using .cpu()
    After that you can convert tensor to numpy using .numpy()
    And you probably know the rest... So basically a.data.cpu().numpy()[0] will give you just the value
"""
  
def criterion(labela, labelb, pred_ab, pred_ba):
    # http://okye062gb.bkt.clouddn.com/2017-10-26-122312.jpg
    assert(labela.size() == pred_ab.size() or labelb.size() == pred_ba.size()), " | Error, predict size is not consisent with label size..."
    B = int(labela.size()[1]/5)
    pred_ab = pred_ab.type('torch.cuda.DoubleTensor')
    pred_ba = pred_ba.type('torch.cuda.DoubleTensor')
    loss = 0
    assert(B == 1), " | Error, bounding box for each grid weird..."
    for i_pair in range(labela.size()[0]):
        for row in range(labela.size()[2]):
            for col in range(labela.size()[3]):
                if labela[i_pair, 0, row, col].data[0]:
                    # Branch 1
                    #print (type(labela[i_pair, 0, row, col]))
                    #print (labela[i_pair, 0, row, col])
                    #print (type(pred_ab[i_pair, 0, row, col]))
                    #print (pred_ab[i_pair, 0, row, col])
                    prob_diff = labela[i_pair, 0, row, col] - pred_ab[i_pair, 0, row, col]
                    loss += torch.mul(prob_diff, prob_diff)
                   # print ("loss stage1", loss)
                    diff_xy = labela[i_pair, 1:3, row, col] - pred_ab[i_pair, 1:3, row, col]
                    loss += torch.sum(torch.mul(diff_xy, diff_xy))
                   # print ("loss stage2", loss)
                    #sqrt_diff_wh = torch.sqrt(labela[i_pair, 3:5, row, col]) - torch.sqrt(pred_ab[i_pair, 3:5, row, col])
                    sqrt_diff_wh = labela[i_pair, 3:5, row, col] - pred_ab[i_pair, 3:5, row, col]
                   # print ("sqrt_diff_wh", sqrt_diff_wh)
                    loss += torch.sum(torch.mul(sqrt_diff_wh, sqrt_diff_wh))
                   # print ("loss stage3", loss)
                    # Branch 2
                    prob_diff = labelb[i_pair, 0, row, col] - pred_ba[i_pair, 0, row, col]
                    loss += torch.mul(prob_diff, prob_diff)
                    diff_xy = labelb[i_pair, 1:3, row, col] - pred_ba[i_pair, 1:3, row, col]
                    loss += torch.sum(torch.mul(diff_xy, diff_xy))
                    #sqrt_diff_wh = torch.sqrt(labelb[i_pair, 3:5, row, col]) - torch.sqrt(pred_ba[i_pair, 3:5, row, col])
                    sqrt_diff_wh = labelb[i_pair, 3:5, row, col] - pred_ba[i_pair, 3:5, row, col]
                    loss += torch.sum(torch.mul(sqrt_diff_wh, sqrt_diff_wh))
    return loss

def cudaftensor2np(var):
    return var.data.cpu().numpy()


class YOLO_loss(torch.nn.Module):
    def forward(self, labela, labelb, pred_ab, pred_ba):
        return criterion(labela, labelb, pred_ab, pred_ba)


if __name__ == "__main__":
    loss = YOLO_loss()
    labela = np.zeros((1, 5, 1, 1))
    labela[0, 0, 0, 0] = 1
    labela[0, 1, 0, 0] = 10
    labela[0, 2, 0, 0] = 20
    labela[0, 3, 0, 0] = 30
    labelb = np.zeros((1, 5, 1, 1))
    labelb[0, 0, 0, 0] = 1
    labelb[0, 1, 0, 0] = 10
    labelb[0, 2, 0, 0] = 20
    labelb[0, 3, 0, 0] = 30
    pred_ab = np.zeros((1, 5, 1, 1))
    pred_ab[0, 0, 0, 0] = 1
    pred_ba = np.zeros((1, 5, 1, 1))

    l = criterion(Variable(torch.from_numpy(labela), requires_grad=True),
                Variable(torch.from_numpy(labelb), requires_grad=True),
                Variable(torch.from_numpy(pred_ab), requires_grad=True),
                Variable(torch.from_numpy(pred_ba), requires_grad=True))

    print (type(l))
    print (l.backward())




