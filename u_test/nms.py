# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from unit import *

def nms(dets, thresh=0.2, score_thresh=0.5):
    """Pure Python NMS baseline."""
    scores = dets[:, 0]
    s_idx = np.where(scores >= score_thresh)[0]
    scores = dets[s_idx, 0]
    x1 = dets[s_idx, 1]
    y1 = dets[s_idx, 2]
    x2 = dets[s_idx, 3]
    y2 = dets[s_idx, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    print (keep)
    print (x1, y1, x2, y2)
    res = np.zeros((len(keep), 5))
    for ii, idx in enumerate(keep):
        res[ii, :] = scores[idx], x1[idx], y1[idx], x2[idx], y2[idx]
    print (res)
    print ("-----------------------")
    return res

if __name__ == "__main__":
    dets = np.array([[   0.99221122 , 209       ,   358,           41,           62,        ], 
                    [   0.6985414 ,  371,          295,           53,           60        ],
                    [   0.5440979 ,  206,          365,           41,           55        ]])
    dets = np.array(mid2mm(dets))
    print (dets)
    print (cal_iou(dets[0, 1:], dets[2, 1:]))
    #print (dets.shape)
    print (nms(dets))
