#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017-10-26 22:09 zq <zq@mclab>
#
# Distributed under terms of the MIT license.

import torch
import numpy as np
import time
import matplotlib.pyplot as plt

from torch.autograd import Variable


__all__ = ['criterion']

"""
How to get value from GPU RAM
    You first need to get tensor out of the variable using .data 
    Then you need to move the tensor to cpu using .cpu()
    After that you can convert tensor to numpy using .numpy()
    And you probably know the rest... So basically a.data.cpu().numpy()[0] will give you just the value
"""

def criterion_branch(label, pred, label_pos, s_prob=1, s_coord=5):
    assert(label.size() == pred.size()), " | Error, predict size is not consisent with label size..."
    B = int(label.size()[1]/5)
    pred = pred.type('torch.cuda.DoubleTensor') 
    loss = Variable(torch.zeros((1)), requires_grad=True).cuda().type('torch.cuda.DoubleTensor')
    lsz = [label.size()[0], label.size()[1], label.size()[2], label.size()[3]]
    lbound = 2+label_pos*4

    # Manually implement softmax
    for i_pair in range(lsz[0]):
        softmax_denominator = torch.sum(torch.exp(pred[i_pair, label_pos, :, :]))
        for row in range(lsz[2]):
            for col in range(lsz[3]):
                pred[i_pair, label_pos, row, col] = torch.exp(pred[i_pair, label_pos, row, col]) / softmax_denominator

    ### SORT DET RESULTS ###

    for i_pair in range(lsz[0]):
        for row in range(lsz[2]):
            for col in range(lsz[3]):
                if label[i_pair, label_pos, row, col].data[0]:
                    #print ("------{}-----".format("label"))
                    #print (label[i_pair, :, row, col].data)
                    #print ("------{}-----".format("pred"))
                    #print (pred[i_pair, :, row, col].data)
                    #iou = cal_iou(label[i_pair, 1:, row, col].data, pred[i_pair, 1:, row, col].data)

                    prob_diff = label[i_pair, label_pos, row, col] - pred[i_pair, label_pos, row, col]
                    diff_xy = label[i_pair, lbound:lbound+2, row, col] - pred[i_pair, lbound:lbound+2, row, col]
                    diff_wh = label[i_pair, lbound+2:lbound+4, row, col] - pred[i_pair, lbound+2:lbound+4, row, col]
                    #sqrt_diff_wh = torch.sqrt(labela[i_pair, 3:5, row, col]) - torch.sqrt(pred_ab[i_pair, 3:5, row, col])

                    loss += torch.mul(prob_diff, prob_diff)*s_prob
                    loss += torch.sum(torch.mul(diff_xy, diff_xy))*s_coord
                    loss += torch.sum(torch.mul(diff_wh, diff_wh))*s_coord*1.414
                else:
                    prob_diff = label[i_pair, label_pos, row, col] - pred[i_pair, label_pos, row, col]
                    loss += torch.mul(prob_diff, prob_diff)*s_prob
    loss = loss/lsz[0]
    #print ("Now loss reaches to {}, pred_size = {}".format(loss, pred_sz[0]))
    return loss


def criterion(label, pred):
    loss1 = criterion_branch(label, pred, 0)
    loss2 = criterion_branch(label, pred, 1)
    loss = (loss1+loss2) / 2
    return loss

def cal_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = (xB - xA) * (yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if boxAArea <= 0.01 or boxBArea <= 0.01 or interArea <= 0.01:
        iou = 0
    else:
        iou = interArea / float(boxAArea + boxBArea - interArea)
    return Variable(torch.FloatTensor([iou])).type('torch.cuda.DoubleTensor')


if __name__ == "__main__":
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




