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

floor = lambda x: math.floor(float(x))
f2s = lambda x: str(float(x))

def detrender(srcdir, imkey, deta_crd, detb_crd, resdir, color="red"):
    im_a = Image.open(osp.join(srcdir, "{:05d}".format(imkey)+"_a.jpg"))
    im_b = Image.open(osp.join(srcdir, "{:05d}".format(imkey)+"_b.jpg"))
    for i_det in deta_crd:
        draw_bbox(im_a, i_det[1:], color)
    for i_det in detb_crd:
        draw_bbox(im_b, i_det[1:], color)
    im_a.save(osp.join(resdir, "{:05d}".format(imkey)+"_render_a.jpg"))
    im_b.save(osp.join(resdir, "{:05d}".format(imkey)+"_render_b.jpg"))

def labelrender(resdir, imkey, gda_crd, gdb_crd, color="green"):
    #print ("imkey", imkey)
    #print ("gda_crd", gda_crd)
    #print ("gdb_crd", gdb_crd)
    im_ra = Image.open(osp.join(resdir, "{:05d}".format(imkey)+"_render_a.jpg"))
    im_rb = Image.open(osp.join(resdir, "{:05d}".format(imkey)+"_render_b.jpg"))
    for i_gd in gda_crd:
        draw_bbox(im_ra, i_gd, color)
    for i_gd in gdb_crd:
        draw_bbox(im_rb, i_gd, color)
    im_ra.save(osp.join(resdir, "{:05d}".format(imkey)+"_render_a.jpg"))
    im_rb.save(osp.join(resdir, "{:05d}".format(imkey)+"_render_b.jpg"))


def draw_bbox(im, bbox, color="red"):
    # bbox should in midx, midy, w, h format
    draw_im = ImageDraw.Draw(im)
    midx, midy, w, h = bbox[:]
    xmin, ymin = floor(midx - w/2.0), floor(midy - h/2.0)
    xmax, ymax = floor(midx + w/2.0), floor(midy + h/2.0)
    draw_im.line([xmin, ymin, xmax, ymin], fill=color)
    draw_im.line([xmin, ymin, xmin, ymax], fill=color)
    draw_im.line([xmax, ymin, xmax, ymax], fill=color)
    draw_im.line([xmin, ymax, xmax, ymax], fill=color)
    del draw_im

def getimsize(im_root, imkey, scale_size=512):
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

def parse_det(label, pred, imkey, imsize, scale_size=512):
    #det result: 1, x, y, w, h -- normalized
    n_bbox = 0
    label_sz = label.size()[:]
    gd_list = []
    for row in range(label_sz[2]):
        for col in range(label_sz[3]):
            if label[0, 0, row, col].data[0]:
                n_bbox += 1
                gd_list.append(label[0, 1:, row, col].data.cpu().numpy().tolist()) 
            if row == 0 and col == 0:
                det = pred[0, :, row, col].data.cpu().numpy()
            else:
                det = np.vstack((det, pred[0, :, row, col].data.cpu().numpy()))

    det_sort = np.sort(det, axis=0)
    s2xB = label_sz[2] * label_sz[3] * 1
    ow, oh = imsize
    sw, sh = float(scale_size)/ow, float(scale_size)/oh

    for i in range(len(det_sort)):
        # detx --> det_midx; dety --> det_midy
        detx, dety, detw, deth = det_sort[i, 1:]
        detx, dety = scale_size*detx, scale_size*dety
        detw, deth = scale_size*detw, scale_size*deth

        orix, oriy = int(detx/sw), int(dety/sh)
        oriw, orih = int(detw/sw), int(deth/sh)

        det_sort[i, 1:] = orix, oriy, oriw, orih

    for i in range(n_bbox):
        gdx, gdy, gdw, gdh = gd_list[i][:]
        gdx,  gdy  = scale_size*gdx, scale_size*gdy
        gdw,  gdh  = scale_size*gdw, scale_size*gdh
        gdx,  gdy  = int(gdx/sw),  int(gdy/sh)
        gdw,  gdh = int(gdw/sw), int(gdh/sh)
        gd_list[i][:] = gdx, gdy, gdw, gdh

    # Extract nbbox results(most high prob)
    det_str = ""
    for i in range(n_bbox):
        det_str += "{:05d}".format(imkey) + " "
        for j in range(label_sz[1]):
            det_str += f2s(det_sort[s2xB-i-1, j]) + " "
        det_str += "\n"
    
    det_list = []
    for i in range(n_bbox):
    #for i in range(s2xB):
        det_list.append(det_sort[s2xB-i-1, :].tolist())

    return det_str, det_list, gd_list


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
