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
from copy import copy
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

"""
LOSS NOTE:

"""


class LossLayer():
    def __init__(self, label, pred, object_scale=1, noobject_scale=0.1, 
                            class_scale=1, coord_scale=5, num_box=1, num_class=2):
        self.object_scale = object_scale
        self.noobject_scale = noobject_scale
        self.class_scale = class_scale
        self.coord_scale = coord_scale
        self.num_class = num_class
        self.diff = np.zeros_like(pred)
        self.lsz = [label.size()[0], label.size()[1], label.size()[2], label.size()[3]]
        self.num_box = (self.lsz[1]-self.num_class)/5
        assert (self.num_box == 2), " | In loss layer, num_box is false!"

    def forward(self):
        avg_iou, avg_cat, avg_allcat, avg_obj, avg_anyobj, count = 0, 0, 0, 0, 0, 0 
        BSZ, ROW, COL = self.lsz[0], self.lsz[2], self.lsz[3]
        # Delta is the differential coefficient
        delta = self.pred[:, :num_box, :, :]
        self.diff = self.noobject_scale * delta     #NOTE
        avg_anyobj balabala
        for i_pair in range(BSZ):
            count_obj = 0
            for row in range(ROW):
                for col in range(COL):
                    if label[i_pair, 0, row, col].data[0]:
                    ### Compute match bounding box of groundtruth
                    #注意在设计中，groundtruth的形式为_|__|____这样的形式，也就是说一张图片上它对应的是一个boundingbox
                    #但是在pred的时候，我们可以设计多个bbox
                    ###
                    truth = copy(label[i_pair, 1+self.num_class:, row, col])
                    truth[0] /= COL     # COL
                    truth[1] /= ROW

                    best_index = 0
                    best_iou = 0
                    best_rmse = 20

                    for i in range(self.num_box):
                        out = copy(pred[i_pair, self.num_box*(self.num_class+1)+4*i:self.num_box*(self.num_class+1)+4*(i+1), row, col])
                        out[0] /= COL
                        out[1] /= ROW
                        out[2] = out[2] ** 2
                        out[3] = out[3] ** 2

                        iou = box_iou(out, truth)
                        rmse = box_rmse(out, truth)
                        # 假如找不到iou不等于0的，就退而求其次找RMSE最小的
                        if (best_iou > 0 or iou > 0):
                            if iou > best_iou:
                                best_iou = iou
                                best_index = i
                        else:
                            if rmse < best_rmse:
                                best_rmse = rmse
                                best_index = i
                    
                    # Forward and backward of category(2)
                    delta = pred[i_pair, self.num_box+self.num_class*best_index:self.num_box+self.num_class*(best_index+1), row, col] - 
                                    label[i_pair, 1:1+self.num_class, row, col]


        
        














### UTILS FOR IOU ###

def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b)

def box_intersection(a, b):
    def overlap(mida, wa, midb, wb):
        l1, l2 = mida - wa/2, midb - wb/2
        left = max(l1, l2)
        r1, r2 = mida + wa/2, midb + wb/2
        right = min(r1, r2)
        return right - left
    w = overlap(a[0], a[2], b[0], b[2])
    h = overlap(a[1], a[3], b[1], b[3])
    if (w <= 0 or h <= 0):
        return 0
    area = w * h
    return area

def box_union(a, b):
    i = box_intersection(a, b)
    u = a[2] * a[3] + b[2] * b[3] - i
    return u

# 均方根误差 root-mean-square error
def box_rmse(a, b):
    res = (a[0] - b[0]) ** 2
    res += (a[1] - b[1]) ** 2
    res += (a[2] - b[2]) ** 2
    res += (a[3] - b[3]) ** 2
    return sqrt(res)




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




