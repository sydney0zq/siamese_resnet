#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017-11-05 22:18 zq <theodoruszq@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""

import numpy as np
    
def cal_iou(boxA, boxB):
    # xmin, ymin, xmax, ymax
    # determine the (x, y)-coordinates of the intersection rectangle
    # and compute the area of intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = (xB - xA) * (yB - yA)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    if boxAArea <= 0 or boxBArea <= 0 or interArea <= 0:
        iou = 0
    else:
        iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def mid2mm(midlist):
    mid_np = np.array(midlist)
    if len(midlist) != 0:
        midx, midy, w, h = mid_np[:, 1], mid_np[:, 2], mid_np[:, 3], mid_np[:, 4]
        minx, miny, maxx, maxy = midx - w/2.0, midy - h/2.0, midx + w/2.0, midy + h/2.0
        mid_np[:, 1], mid_np[:, 2], mid_np[:, 3], mid_np[:, 4] = minx, miny, maxx, maxy
    return mid_np.tolist()
