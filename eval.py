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
import os.path as osp

from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler 
import xml.etree.ElementTree as ET

from model.model import DiffNetwork
from data.dataset import Pair_Dataset
from loss import criterion
import os.path as osp

from utils import getimsize, detrender, labelrender, parse_det, parse_gd

def evaluate(args):
    ### DATA ###
    dataclass = Pair_Dataset(args.test_dir, train=False)
    imkey_list = dataclass.imkey_list
    dataloader = {}
    dataloader["test"] = DataLoader(dataclass,
                                  1, 
                                  shuffle=False, 
                                  num_workers=args.num_workers)

    ### LOAD MODEL ###
    if osp.exists(args.model):
        model_weights = torch.load(args.model)
        model = DiffNetwork()
        model.load_state_dict(model_weights)
        if args.cuda:
            model = model.cuda()
    else:
        raise IOError

    ### START TO EUALUATE ###
    tic = time.time()
    running_loss = 0
    fa = open(args.deta_fn, "w")
    fb = open(args.detb_fn, "w")

    for ii, (index, im_a, im_b, label) in enumerate(dataloader["test"]):
        inp_a, inp_b = Variable(im_a), Variable(im_b)
        label = Variable(label)
        if args.cuda:
            inp_a, inp_b = inp_a.cuda(), inp_b.cuda()
            label = label.cuda()
        pred = model(inp_a, inp_b)
        loss = criterion(label, pred)
        running_loss += loss.data[0]
        imkey = int(imkey_list[index[0]])
        imsize = getimsize(args.test_dir, imkey)

        # deta_crd and gda_crd are both (midx, midy, w, h)
        gda_crd, gdb_crd = parse_gd(label, imsize, 1), parse_gd(label, imsize, 2)
        deta_str, deta_crd = parse_det(pred, imkey, imsize, 1)
        detb_str, detb_crd = parse_det(pred, imkey, imsize, 2)
        #print (ave_iou(deta_crd, gda_crd))
        #print (ave_iou(detb_crd, gdb_crd))

        # Write str to det files
        fa.write(deta_str)
        fb.write(detb_str)
        
        # Render predictions
        detrender(args.test_dir, imkey, deta_crd, detb_crd, args.desdir)
        labelrender(args.test_dir, args.desdir, imkey, gda_crd, gdb_crd)

    fa.close()
    fb.close()
    print (" | -- Eval Ave Loss {:.2f}".format(running_loss/(ii+1))) 
    print (" | Time consuming: {:.2f}s".format(time.time()-tic))

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, default="./data/test")
    parser.add_argument('--num_workers', type=int, default=8,
                            help="Number of data loading threads.")
    parser.add_argument('--no_cuda', action='store_true', default=False,
                            help="Disable CUDA training.")
    parser.add_argument('--model', type=str, default="./model_best.pth.tar", 
                            help="Give a model to test.")
    parser.add_argument('--deta_fn', type=str, default="./result/det_a.txt", 
                            help="Detection result filename of image a.")
    parser.add_argument('--detb_fn', type=str, default="./result/det_b.txt", 
                            help="Detection result filename of image b.")
    parser.add_argument('--desdir', type=str, default="./result",
                            help="Rendered image directory.")

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args


def u_test(args):
    dataloader = {}
    dataclass = Pair_Dataset(args.test_dir, train=False)
    imkey_list = dataclass.imkey_list
    dataloader["test"] = DataLoader(dataclass, 1, shuffle=False, num_workers=args.num_workers)
    for ii, (index, im_a, im_b, label) in enumerate(dataloader["test"]):
        label = Variable(label)
        imkey = int(imkey_list[index[0]])
        imsize = getimsize(args.test_dir, imkey)
        # deta_crd and gda_crd are both (midx, midy, w, h)
        print (label)
        gda_crd, gdb_crd = parse_gd(label, imsize, 1), parse_gd(label, imsize, 2)
        print (gda_crd, gdb_crd)

        labelrender(args.desdir, imkey, gda_crd, gdb_crd)



if __name__ == "__main__":
    args = parse()
    evaluate(args)
    #u_test(args)


