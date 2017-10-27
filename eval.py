#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017-10-25 21:09 zq <zq@mclab>
#
# Distributed under terms of the MIT license.

"""
Evaluate our best model.
"""

import argparse
import torch
import random
import numpy as np
import time
import matplotlib.pyplot as plt

from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler 
import xml.etree.ElementTree as ET

from model import SiameseBranchNetwork
from data.dataset import Pair_Dataset
from yolo_loss import criterion
import os.path as osp


def evaluate(args):
    ### DATA ###
    dataclass = Pair_Dataset(args.test_dir, train=False)
    imkey_list = dataclass.imkey_list
    dataloader = {}
    dataloader["test"] = DataLoader(dataclass,
                                  1, 
                                  shuffle=True, 
                                  num_workers=args.num_workers)

    ### LOAD MODEL ###
    if osp.exists(args.model):
        model_weights = torch.load(args.model)
        model = SiameseBranchNetwork()
        model.load_state_dict(model_weights)
        if args.cuda:
            model = model.cuda()
    else:
        raise IOError


    ### START TO EUALUATE ###
    tic = time.time()
    running_loss = 0
    fa = open(args.branch_a, "w")
    fb = open(args.branch_b, "w")

    for ii, (index, diff_ab, diff_ba, labela, labelb) in enumerate(dataloader["test"]):
        inp_ab, inp_ba = Variable(diff_ab), Variable(diff_ba)
        labela, labelb = Variable(labela), Variable(labelb)
        if args.cuda:
            inp_ab, inp_ba = inp_ab.cuda(), inp_ba.cuda()
            labela, labelb = labela.cuda(), labelb.cuda()
        pred_ab, pred_ba = model(inp_ab, inp_ba)
        loss = criterion(labela, labelb, pred_ab, pred_ba)

        imsize = getimsize(imkey_list[index[0]], args.test_dir)
        fa.write(parse_det(labela, pred_ab, imkey_list[index[0]], imsize))
        fb.write(parse_det(labelb, pred_ba, imkey_list[index[0]], imsize))
        print (" | Eval {} Loss {:.2f}".format(ii+1, loss.data[0])) 
        running_loss += loss.data[0]
    fa.close()
    fb.close()
    print (" | -- Eval Ave Loss {:.2f}".format(running_loss/(ii+1))) 
    print (" | Time consuming: {:.2f}s".format(time.time()-tic))

def f2s(x):
    return str(float(x))
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def parse_det(label, pred, imkey, imsize, scale_size=512):
    n_bbox = 0
    for row in range(label.size()[2]):
        for col in range(label.size()[3]):
            if label[0, 0, row, col].data[0]:
                n_bbox += 1
            if row == 0 and col == 0:
                det = pred[0, :, row, col].data.cpu().numpy()
            else:
                det = np.vstack((det, pred[0, :, row, col].data.cpu().numpy()))
    
    det_sort = np.sort(det, axis=0)
    # 1, x, y, w, h -- normalized
    s2xB = label.size()[2] * label.size()[3] * 1
    ow, oh = imsize
#    det_sort[:, 0] = softmax(det_sort[:, 0])

    for i in range(len(det_sort)):
        detx, dety, detw, deth = det_sort[i, 1:]
        detx, dety = scale_size*detx, scale_size*dety
        detw, deth = scale_size*detw, scale_size*deth
        sw, sh = float(scale_size)/ow, float(scale_size)/oh
        orix, oriy = int(detx/sw), int(dety/sh)
        oriw, orih = int(detw/sw), int(deth/sh)
        det_sort[i, 1:] = orix, oriy, oriw, orih

    # Extract nbbox results
    det_str = ""
    for i in range(n_bbox):
        det_str += "{:05d}".format(imkey) + " "
        for j in range(label.size()[1]):
            det_str += f2s(det_sort[s2xB-i-1, j]) + " "
        det_str += "\n"

    return det_str

def getimsize(imkey, im_root, scale_size=512):
    if osp.exists(osp.join(im_root, "{:05d}".format(imkey)+"_a.xml")):
        xmlpath = osp.join(im_root, "{:05d}".format(imkey)+"_a.xml")
    else:
        xmlpath = osp.join(im_root, "{:05d}".format(imkey)+"_b.xml")
    tree = ET.parse(xmlpath)
    im_size = tree.findall("size")[0]
    ow = int(im_size.find("width").text)
    oh = int(im_size.find("height").text)
    return (ow, oh)


def parse():
    parser = argparse.ArgumentParser()
    ### DATA ###
    parser.add_argument('--test_dir', type=str, default="./data/test")
    parser.add_argument('--num_workers', type=int, default=8,
                            help="Number of data loading threads.")
    parser.add_argument('--no_cuda', action='store_true', default=False,
                            help="Disable CUDA training.")
    parser.add_argument('--model', type=str, default="./model_test.pth.tar", 
                            help="Give a model to test.")
    parser.add_argument('--branch_a', type=str, default="./result/det_a.txt", 
                            help="Branch a detection result.")
    parser.add_argument('--branch_b', type=str, default="./result/det_b.txt", 
                            help="Branch b detection result.")

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    #print('Args: {}'.format(args))
    return args


if __name__ == "__main__":
    args = parse()
    evaluate(args)




