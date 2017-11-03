#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017-10-25 21:14 zq <zq@mclab>
#
# Distributed under terms of the MIT license.

"""
Testing for dataset.py.
"""
import PIL
from PIL import Image
from PIL import ImageDraw
import math
import numpy as np
import os.path as osp
import torch
import xml.etree.ElementTree as ET
from torchvision import transforms
from dataset import Pair_Dataset
from torch.utils.data import DataLoader

floor = lambda x: math.floor(float(x))
f2s = lambda x: str(float(x))

def labelrender(im_ra, im_rb, imkey, gda_crd, gdb_crd, color="green"):
    for i_gd in gda_crd:
        draw_bbox(im_ra, i_gd, color)
    for i_gd in gdb_crd:
        draw_bbox(im_rb, i_gd, color)
    im_ra.save(osp.join("./u_test", imkey+"_render_a.jpg"))
    im_rb.save(osp.join("./u_test", imkey+"_render_b.jpg"))


def getimsize(im_root, imkey):
    assert (type(imkey) == type("")), " | Imkey should be string type..."
    if osp.exists(osp.join(im_root, imkey+"_a.xml")):
        xmlpath = osp.join(im_root, imkey+"_a.xml")
    else:
        xmlpath = osp.join(im_root, imkey+"_b.xml")
    tree = ET.parse(xmlpath)
    im_size = tree.findall("size")[0]
    ow = int(im_size.find("width").text)
    oh = int(im_size.find("height").text)
    return (ow, oh)

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

def parse_gd(label, imsize, pairwise, scale_size=512):
    """ NO jitter data augmentation when testing. """
    ROW, COL = label.size()[2:]
    gd_list = []
    n_bbox = 0
    ow, oh = imsize
    sx, sy = scale_size*1.0/ow, scale_size*1.0/oh
    for row in range(ROW):
        for col in range(COL):
            # Generate groundtruth list
            # We only have one instance each time at evalution stage
            if label[0, pairwise, row, col]:
                n_bbox += 1
                x = (label[0, 3, row, col] + col) / COL
                y = (label[0, 4, row, col] + row) / ROW
               # x, y = label[0, 3:5, row, col]
                w, h = label[0, 5:, row, col]
                gd_list.append([x, y, w, h])

    for i in range(n_bbox):
        gdx, gdy, gdw, gdh = gd_list[i][:]
        gdx,  gdy  = ow*gdx*sx, oh*gdy*sy
        gdw,  gdh  = ow*gdw*sx, oh*gdh*sy
        gd_list[i][:] = gdx, gdy, gdw, gdh

    return gd_list

def parse_gd(label, imsize, pairwise, scale_size=512):
    """ NO jitter data augmentation when testing. """
    ROW, COL = label.size()[2:]
    gd_list = []
    n_bbox = 0
    ow, oh = imsize
    sx, sy = scale_size*1.0/ow, scale_size*1.0/oh
    for row in range(ROW):
        for col in range(COL):
            # Generate groundtruth list
            # We only have one instance each time at evalution stage
            if label[0, pairwise, row, col]:
                n_bbox += 1
                x = (label[0, 3, row, col] + col) / COL
                y = (label[0, 4, row, col] + row) / ROW
               # x, y = label[0, 3:5, row, col]
                w, h = label[0, 5:, row, col]
                gd_list.append([x, y, w, h])

    for i in range(n_bbox):
        gdx, gdy, gdw, gdh = gd_list[i][:]
        gdx,  gdy  = scale_size*gdx*sx, scale_size*gdy*sy
        gdw,  gdh  = scale_size*gdw*sx, scale_size*gdh*sy
        gd_list[i][:] = gdx, gdy, gdw, gdh

    return gd_list

def u_test():
    utest_dir = "./u_test"
    dataset_dir = "./test"
    utest_dataset = Pair_Dataset(dataset_dir, test=True)
    utestloader = DataLoader(utest_dataset, batch_size = 1, shuffle=True, num_workers=1)
    imkeys = utest_dataset.imkey_list

    for ii, (index, im_a, im_b, label) in enumerate(utestloader):
        imkey = imkeys[index[0]]
        """
        if imkey == "00583":
            imap = "./test/" + imkey + "_a.jpg"
            imbp = "./test/" + imkey + "_b.jpg"
            ima = Image.open(imap)
            imb = Image.open(imbp)
            print (imkey)
            imsize = getimsize(dataset_dir, imkey)
            gda_crd, gdb_crd = parse_gd_test(label, imsize, 1), parse_gd_test(label, imsize, 2)
            labelrender(ima, imb, imkey, gda_crd, gdb_crd)
            print (gda_crd)
            print (gdb_crd)
        """
        """ LOAD IMAGES TO 512x512 size"""
        #im_a = im_a.view(-1, 512, 512)  # Should be 3x512x512
        #imapil = transforms.ToPILImage()(im_a)
        #im_b = im_b.view(-1, 512, 512)
        #imbpil = transforms.ToPILImage()(im_b)

        # DRAW TRANSFORMED IMAGE AND LABEL
        #imsize = getimsize(dataset_dir, imkey)
        #gda_crd, gdb_crd = parse_gd(label, imsize, 1), parse_gd(label, imsize, 2)
        #labelrender(imapil, imbpil, imkey, gda_crd, gdb_crd)
        if imkey == "00557":
            #print (label)
            pass
        
    


u_test()




















