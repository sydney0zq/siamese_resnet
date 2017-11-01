#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017-10-25 21:14 zq <zq@mclab>
#
# Distributed under terms of the MIT license.

"""
Utils.
"""
import PIL
from PIL import Image
from PIL import ImageDraw
import math
import numpy as np
import os.path as osp
import xml.etree.ElementTree as ET
from nms import nms

floor = lambda x: math.floor(float(x))
f2s = lambda x: str(float(x))


# label is 5x7x7
# pred  is 7x7x7
def parse_gd(label, imsize, pairwise):
    ROW, COL = label.size()[2:]
    gd_list = []
    n_bbox = 0
    ow, oh = imsize
    label = label.data.cpu().numpy()
    for row in range(ROW):
        for col in range(COL):
            # Generate groundtruth list
            # We only have one instance each time at evalution stage
            if label[0, pairwise, row, col]:
                n_bbox += 1
                x = (label[0, 3, row, col] + col) / COL
                y = (label[0, 4, row, col] + row) / ROW
                #x, y = label[0, 3:5, row, col]
                w, h = label[0, 5:, row, col]
                gd_list.append([x, y, w, h])

    for i in range(n_bbox):
        gdx, gdy, gdw, gdh = gd_list[i][:]
        gdx,  gdy  = ow*gdx, oh*gdy
        gdw,  gdh  = ow*gdw, oh*gdh
        gd_list[i][:] = gdx, gdy, gdw, gdh

    return gd_list

def parse_det(pred, imkey, imsize, pairwise):
    #det result: prob_obj, pa, pb, x, y, w, h -- normalized
    ROW, COL = pred.size()[2:]
    det = np.zeros((1, 5))
    ow, oh = imsize

    pred = pred.data.cpu().numpy()

    for row in range(ROW):
        for col in range(COL):
            if row == 0 and col == 0:
                det[0, 0] = pred[0, 0, row, col] * pred[0, pairwise, row, col]
                det[0, 1:] = pred[0, 3:, row, col]
                det = det.reshape(1, 5)
            else:
                temp = np.zeros((1, 5))
                temp[0, 0] = pred[0, 0, row, col] * pred[0, pairwise, row, col]
                temp[0, 1:] = pred[0, 3:, row, col]
                temp = temp.reshape(1, 5)
                det = np.vstack((det, temp))

    det_len = det.shape[0]

    for i in range(det_len):
        # detx means det_midx; dety means det_midy
        # Recover to origin size
        detx, dety, detw, deth = det[i, 1:]
        orix, oriy = int(detx*ow), int(dety*oh)
        oriw, orih = int(detw*ow), int(deth*oh)
        det[i, 1:] = orix, oriy, oriw, orih
    
    det_list = det[nms(det)]
    det_len = len(det_list)

    det_str = ""
    for i in range(det_len):
        det_str += "{:05d}".format(imkey) + " "
        for j in range(5):
            det_str += f2s(det_list[i][j]) + " "
        det_str += "\n"

    return det_str, det_list


### RENDER AREA ###
""" All Renders receive list format result """
def detrender(srcdir, imkey, deta_crd, detb_crd, resdir, color="red"):
    im_a = Image.open(osp.join(srcdir, "{:05d}".format(imkey)+"_a.jpg"))
    im_b = Image.open(osp.join(srcdir, "{:05d}".format(imkey)+"_b.jpg"))
    for i_det in deta_crd:
        draw_bbox(im_a, i_det[1:], color)
    for i_det in detb_crd:
        draw_bbox(im_b, i_det[1:], color)
    im_a.save(osp.join(resdir, "{:05d}".format(imkey)+"_render_a.jpg"))
    im_b.save(osp.join(resdir, "{:05d}".format(imkey)+"_render_b.jpg"))

def labelrender(srcdir, resdir, imkey, gda_crd, gdb_crd, color="green"):
    if osp.exists(osp.join(resdir, "{:05d}".format(imkey)+"_render_a.jpg")):
        im_ra = Image.open(osp.join(resdir, "{:05d}".format(imkey)+"_render_a.jpg"))
        im_rb = Image.open(osp.join(resdir, "{:05d}".format(imkey)+"_render_b.jpg"))
    else:
        im_ra = Image.open(osp.join(srcdir, "{:05d}".format(imkey)+"_a.jpg"))
        im_rb = Image.open(osp.join(srcdir, "{:05d}".format(imkey)+"_b.jpg"))

    for i_gd in gda_crd:
        draw_bbox(im_ra, i_gd, color)
    for i_gd in gdb_crd:
        draw_bbox(im_rb, i_gd, color)
    im_ra.save(osp.join(resdir, "{:05d}".format(imkey)+"_render_a.jpg"))
    im_rb.save(osp.join(resdir, "{:05d}".format(imkey)+"_render_b.jpg"))

def draw_bbox(im, bbox, color="red"):
    # bbox should in midx, midy, w, h list format
    draw_im = ImageDraw.Draw(im)
    midx, midy, w, h = bbox[:]
    xmin, ymin = floor(midx - w/2.0), floor(midy - h/2.0)
    xmax, ymax = floor(midx + w/2.0), floor(midy + h/2.0)
    draw_im.line([xmin, ymin, xmax, ymin], fill=color)
    draw_im.line([xmin, ymin, xmin, ymax], fill=color)
    draw_im.line([xmax, ymin, xmax, ymax], fill=color)
    draw_im.line([xmin, ymax, xmax, ymax], fill=color)
    del draw_im

def getimsize(im_root, imkey):
    assert (type(imkey) == type(1)), " | Error: imkey shold be an integer..."
    if osp.exists(osp.join(im_root, "{:05d}".format(imkey)+"_a.xml")):
        xmlpath = osp.join(im_root, "{:05d}".format(imkey)+"_a.xml")
    else:
        xmlpath = osp.join(im_root, "{:05d}".format(imkey)+"_b.xml")
    tree = ET.parse(xmlpath)
    im_size = tree.findall("size")[0]
    ow = int(im_size.find("width").text)
    oh = int(im_size.find("height").text)
    return (ow, oh)

### CALCULATE IOU ###
"""
A criterion to calculate the score we get.
You should never apply it to loss function, beacause it has many dependices.
"""

def cal_iou(boxA, boxB):
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


def ave_iou(detlist, gdlist):
    # The two list are both in midx, midy, w, h in original size
    ave_iou = np.zeros((len(gdlist)))
    for ii, i_gd in enumerate(gdlist):
        for jj, i_det in enumerate(detlist):
            ave_iou[ii] += cal_iou(i_gd, i_det)
        ave_iou[ii] = ave_iou[ii] / (jj+1.0)
    return ave_iou


if __name__ == "__main__":
    im = Image.open("./data/test/00589_a.jpg")
    draw_bbox(im, [100, 400, 70, 90])
    im.save("test.jpg")
