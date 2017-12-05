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
from PIL import ImageFont
from PIL import ImageDraw
import math
import numpy as np
import os.path as osp
import xml.etree.ElementTree as ET
from nms import nms

floor = lambda x: math.floor(float(x))
f2s = lambda x: str(float(x))


### HIGH LEVEL RENDER TOOLS ###
def render_orim(args, imkey, label, pred):
    font = ImageFont.truetype(args.fontfn, 12)
    imsize = getimsize(args.test_dir, imkey)
    # deta_crd and gda_crd are both (midx, midy, w, h) on scaled images
    gda_crd, gdb_crd = parse_gd(label, imsize, 1), parse_gd(label, imsize, 2)
    deta_crd = parse_det(pred, imkey, imsize, 1)
    detb_crd = parse_det(pred, imkey, imsize, 2)
    
    # Scale bbox
    gda_crd, gdb_crd = scale_trans(gda_crd, imsize), scale_trans(gdb_crd, imsize)
    deta_crd, detb_crd = scale_trans(deta_crd, imsize), scale_trans(detb_crd, imsize)
    deta_str, detb_str = bbox2str(deta_crd, imkey, 1), bbox2str(detb_crd, imkey, 2)

    # Render on images
    if args.render == 1:
        im_a_path = osp.join(args.test_dir, imkey + "_a.jpg")
        im_b_path = osp.join(args.test_dir, imkey + "_b.jpg")
        im_a, im_b = Image.open(im_a_path), Image.open(im_b_path)
        labelrender_t(im_a, im_b, args.desdir, imkey, gda_crd, gdb_crd)
        detrender_t(im_a, im_b, args.desdir, imkey, deta_crd, detb_crd, font)
    return deta_str, detb_str

def render_wo_gd(args, imkey, pred):
    font = ImageFont.truetype(args.fontfn, 12)
    imsize = getimsize(args.test_dir, imkey)
    # deta_crd and gda_crd are both (midx, midy, w, h) on scaled images
    deta_crd = parse_det(pred, imkey, imsize, 1)
    detb_crd = parse_det(pred, imkey, imsize, 2)
    
    # Scale bbox
    deta_crd, detb_crd = scale_trans(deta_crd, imsize), scale_trans(detb_crd, imsize)
    deta_str, detb_str = bbox2str(deta_crd, imkey, 1), bbox2str(detb_crd, imkey, 2)

    # Render on images
    if args.render == 1:
        im_a_path = osp.join(args.test_dir, imkey + "_a.jpg")
        im_b_path = osp.join(args.test_dir, imkey + "_b.jpg")
        im_a, im_b = Image.open(im_a_path), Image.open(im_b_path)
        detrender_t(im_a, im_b, args.desdir, imkey, deta_crd, detb_crd, font)
    return deta_str, detb_str


# label is 7x7x7
# pred  is 7x7x7
""" NOTE sx and sy is 1 and 1 """
def parse_gd(label, imsize, pairwise, scale_size=512):
    ROW, COL = label.size()[2:]
    gd_list = []
    n_bbox = 0
    ow, oh = imsize
    label = label.data.cpu().numpy()
    sx, sy = 1, 1

    for row in range(ROW):
        for col in range(COL):
            # We only have one instance each time at evalution stage
            if label[0, pairwise, row, col]:
                n_bbox += 1
                x = (label[0, 3, row, col] + col) / COL
                y = (label[0, 4, row, col] + row) / ROW
                w, h = label[0, 5:, row, col]
                gd_list.append([x, y, w, h])

    for i in range(n_bbox):
        gdx, gdy, gdw, gdh = gd_list[i][:]
        ### USE FOLLOW CODE TO RECOVER TO ORIGIN SIZE, WITH BUGGY###
        gdx,  gdy  = scale_size*gdx*sx, scale_size*gdy*sy
        gdw,  gdh  = scale_size*gdw*sx, scale_size*gdh*sy
        gd_list[i][:] = 1, gdx, gdy, gdw, gdh

    return gd_list

def parse_det(pred, imkey, imsize, pairwise, scale_size=512):
    #det result: prob_obj, pa, pb, x, y, w, h -- normalized
    ROW, COL = pred.size()[2:]
    det = np.zeros((1, 5))
    ow, oh = imsize
    sx, sy = 1, 1

    pred = pred.data.cpu().numpy()
    for row in range(ROW):
        for col in range(COL):
            if row == 0 and col == 0:
                det[0, 0] = pred[0, 0, row, col] * pred[0, pairwise, row, col]
                det[0, 1] = (pred[0, 3, row, col] + col) / COL
                det[0, 2] = (pred[0, 4, row, col] + row) / ROW
                det[0, 3:] = pred[0, 5:, row, col]
                det = det.reshape(1, 5)
            else:
                temp = np.zeros((1, 5))
                temp[0, 0] = pred[0, 0, row, col] * pred[0, pairwise, row, col]
                temp[0, 1] = (pred[0, 3, row, col] + col) / COL
                temp[0, 2] = (pred[0, 4, row, col] + row) / ROW
                temp[0, 3:] = pred[0, 5:, row, col]
                temp = temp.reshape(1, 5)
                det = np.vstack((det, temp))

    for i in range(det.shape[0]):
        # detx means det_midx; dety means det_midy
        detx, dety, detw, deth = det[i, 1:]
        """USE THE FOLLOW CODE TO RECOVER FROM ORIGIN IMAGES, WITH BUGGY"""
        #orix, oriy = int(detx*ow*sx), int(dety*oh*sy)
        #oriw, orih = int(detw*ow*sx), int(deth*oh*sy)
        orix, oriy = int(detx*scale_size*sx), int(dety*scale_size*sy)
        oriw, orih = int(detw*scale_size*sx), int(deth*scale_size*sy)
        det[i, 1:] = orix, oriy, oriw, orih
    
    # NOTE: We need to nms
    det_mm = np.array(mid2mm(det))
    det_list = mm2mid( nms(det_mm) )
    #det_list = det
    return det_list


### RENDER AREA ###
""" All Renders receive list format result """
def detrender(srcdir, desdir, imkey, deta_crd, detb_crd, font, color="#00ff00"):
    im_a = Image.open(osp.join(srcdir, imkey+"_a.jpg"))
    im_b = Image.open(osp.join(srcdir, imkey+"_b.jpg"))
    for i_det in deta_crd:
        draw_bbox(im_a, i_det[1:], color)
        draw_prob(im_a, i_det, font, color)
    for i_det in detb_crd:
        draw_bbox(im_b, i_det[1:], color)
        draw_prob(im_b, i_det, font, color)
    im_a.save(osp.join(desdir, imkey+"_render_a.jpg"))
    im_b.save(osp.join(desdir, imkey+"_render_b.jpg"))

def detrender_t(im_a, im_b, desdir, imkey, deta_crd, detb_crd, font, color="#00ff00"):
    for i_det in deta_crd:
        draw_bbox(im_a, i_det[1:], color)
        draw_prob(im_a, i_det, font, color)
    for i_det in detb_crd:
        draw_bbox(im_b, i_det[1:], color)
        draw_prob(im_b, i_det, font, color)
    im_a.save(osp.join(desdir, imkey+"_render_a.jpg"))
    im_b.save(osp.join(desdir, imkey+"_render_b.jpg"))

def labelrender(srcdir, desdir, imkey, gda_crd, gdb_crd, color="#ff0000"):
    """ USE FOLLOW CODE TO RENDER FROM ORIGIN IMAGES, WITH BUGGY
    if osp.exists(osp.join(desdir, imkey+"_render_a.jpg")):
        im_ra = Image.open(osp.join(desdir, imkey+"_render_a.jpg"))
        im_rb = Image.open(osp.join(desdir, imkey+"_render_b.jpg"))
    else:
        im_ra = Image.open(osp.join(srcdir, imkey+"_a.jpg"))
        im_rb = Image.open(osp.join(srcdir, imkey+"_b.jpg"))
    """

    for i_gd in gda_crd:
        draw_bbox(im_ra, i_gd[1:], color)
    for i_gd in gdb_crd:
        draw_bbox(im_rb, i_gd[1:], color)
    im_ra.save(osp.join(desdir, imkey+"_render_a.jpg"))
    im_rb.save(osp.join(desdir, imkey+"_render_b.jpg"))

def labelrender_t(im_ra, im_rb, desdir, imkey, gda_crd, gdb_crd, color="#ff0000"):
    for i_gd in gda_crd:
        draw_bbox(im_ra, i_gd[1:], color)
    for i_gd in gdb_crd:
        draw_bbox(im_rb, i_gd[1:], color)
    im_ra.save(osp.join(desdir, imkey+"_render_a.jpg"))
    im_rb.save(osp.join(desdir, imkey+"_render_b.jpg"))

def draw_bbox(im, bbox, color="#00ff00"):
    # bbox should in midx, midy, w, h list format
    draw_im = ImageDraw.Draw(im)
    midx, midy, w, h = bbox[:]
    xmin, ymin = floor(midx - w/2.0), floor(midy - h/2.0)
    xmax, ymax = floor(midx + w/2.0), floor(midy + h/2.0)
    draw_im.line([xmin, ymin, xmax, ymin], width=2, fill=color)
    draw_im.line([xmin, ymin, xmin, ymax], width=2, fill=color)
    draw_im.line([xmax, ymin, xmax, ymax], width=2, fill=color)
    draw_im.line([xmin, ymax, xmax, ymax], width=2, fill=color)
    del draw_im

def draw_prob(im, bbox, font, color="#00ff00"):
    im_draw = ImageDraw.Draw(im)
    prob_str = "{:.3f}".format(float(bbox[0]))
    topleftx = bbox[1] - bbox[3]/2.0
    toplefty = bbox[2] - bbox[4]/2.0
    im_draw.text((topleftx, toplefty), prob_str, font=font, fill=color)
    del im_draw

def getimsize(im_root, imkey):
    assert (type(imkey) == type("")), " | Error: imkey shold be string type..."
    xmlpath = ""
    if osp.exists(osp.join(im_root, imkey+"_a.xml")):
        xmlpath = osp.join(im_root, imkey+"_a.xml")
    elif osp.exists(osp.join(im_root, imkey+"_b.xml")):
        xmlpath = osp.join(im_root, imkey+"_b.xml")
    if xmlpath != "":
        tree = ET.parse(xmlpath)
        im_size = tree.findall("size")[0]
        ow = int(im_size.find("width").text)
        oh = int(im_size.find("height").text)
    else:
        im = Image.open(osp.join(im_root, imkey+"_a.jpg"))
        ow, oh = im.size
    return (ow, oh)

def scale_trans(bbox, imsize, scale_size=512):
    # gd and det should be Nx5 format
    ow, oh = imsize
    sx, sy = ow*1.0/scale_size, oh*1.0/scale_size
    bbox_np = np.array(bbox)
    if len(bbox) != 0:
        bbox_np[:, 1] = bbox_np[:, 1] * sx
        bbox_np[:, 3] = bbox_np[:, 3] * sx
        bbox_np[:, 2] = bbox_np[:, 2] * sy
        bbox_np[:, 4] = bbox_np[:, 4] * sy
    return bbox_np.tolist()

def bbox2str(det_list, imkey, pairwise):
    # det_list in the format of midx, midy, w, h
    det_str = ""
    det_len = len(det_list)
    det = mid2mm(det_list)
    for i in range(det_len):
        det_str += imkey + "_" + chr(ord('a')+pairwise-1) + " "
        for j in range(5):
            det_str += f2s(det[i][j]) + " "
        det_str += "\n"
    return det_str

def mid2mm(midlist):
    mid_np = np.array(midlist)
    if len(midlist) != 0:
        midx, midy, w, h = mid_np[:, 1], mid_np[:, 2], mid_np[:, 3], mid_np[:, 4]
        minx, miny, maxx, maxy = midx - w/2.0, midy - h/2.0, midx + w/2.0, midy + h/2.0
        mid_np[:, 1], mid_np[:, 2], mid_np[:, 3], mid_np[:, 4] = minx, miny, maxx, maxy
    return mid_np.tolist()
    
def mm2mid(mmlist):
    mm_np = np.array(mmlist)
    if len(mmlist) != 0:
        minx, miny, maxx, maxy = mm_np[:, 1], mm_np[:, 2], mm_np[:, 3], mm_np[:, 4]
        midx, midy, w,    h    = (minx+maxx)/2.0, (miny+maxy)/2.0, (maxx-minx), (maxy-miny)
        mm_np[:, 1], mm_np[:, 2], mm_np[:, 3], mm_np[:, 4] = midx, midy, w, h
    return mm_np.tolist()

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
    a = [[0.3, 10, 20, 50, 90], [0.6, 20, 50, 40, 100]]
    b = [[0.4, 10, 20, 50, 90], [0.5, 20, 50, 40, 100]]
    print (a)
    print (mm2mid(a))
