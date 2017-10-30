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

def criterion_branch(label, pred, s_coord=5):
    assert(label.size() == pred.size()), " | Error, predict size is not consisent with label size..."
    B = int(label.size()[1]/5)
    pred = pred.type('torch.cuda.DoubleTensor') 
    loss = Variable(torch.zeros((1)), requires_grad=True).cuda().type('torch.cuda.DoubleTensor')
    pred_sz = [label.size()[0], label.size()[1], label.size()[2], label.size()[3]]

    # Manually implement softmax
    """
    for i_pair in range(pred_sz[0]):
        softmax_denominator = torch.sum(torch.exp(pred[i_pair, 0, :, :]))
        #print (softmax_denominator)
        for row in range(pred_sz[2]):
            for col in range(pred_sz[3]):
                pred[i_pair, 0, row, col] = torch.exp(pred[i_pair, 0, row, col]) / softmax_denominator
    """

    for i_pair in range(pred_sz[0]):
        for row in range(pred_sz[2]):
            for col in range(pred_sz[3]):
                if label[i_pair, 0, row, col].data[0]:
                    #print ("------{}-----".format("label"))
                    #print (label[i_pair, :, row, col].data)
                    #print ("------{}-----".format("pred"))
                    #print (pred[i_pair, :, row, col].data)
                    #iou = cal_iou(label[i_pair, 1:, row, col].data, pred[i_pair, 1:, row, col].data)

                    #pred_loss = pred[i_pair, 0, row, col] - iou_loss

                    diff_xy = label[i_pair, 1:3, row, col] - pred[i_pair, 1:3, row, col]
                    diff_wh = label[i_pair, 3:5, row, col] - pred[i_pair, 3:5, row, col]
                    #sqrt_diff_wh = torch.sqrt(labela[i_pair, 3:5, row, col]) - torch.sqrt(pred_ab[i_pair, 3:5, row, col])

                    #loss += torch.abs(prob_diff) * s_prob
                    #loss += pred_loss
                    loss += torch.sum(torch.abs(diff_xy))*s_coord*1.414
                    loss += torch.sum(torch.abs(diff_wh))*s_coord
                   # print (prob_diff)
                   # loss += torch.mul(prob_diff, prob_diff)*s_prob
                   # print ("loss stage1", loss)
                   # loss += torch.sum(torch.mul(diff_xy, diff_xy))*s_coord
                   # print ("loss stage2", loss)
                   # loss += torch.sum(torch.mul(sqrt_diff_wh, sqrt_diff_wh))*s_coord*1.414
    loss = loss/pred_sz[0]
    #print ("Now loss reaches to {}, pred_size = {}".format(loss, pred_sz[0]))
    return loss

def criterion(labela, labelb, pred_ab, pred_ba):
    # http://okye062gb.bkt.clouddn.com/2017-10-26-122312.jpg
    loss1 = criterion_branch(labela, pred_ab)
    loss2 = criterion_branch(labelb, pred_ba)
    return (loss1+loss2) / 2


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




