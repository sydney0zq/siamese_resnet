#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017-10-25 21:09 zq <zq@mclab>
#
# Distributed under terms of the MIT license.

"""
Evaluate our best model without groundtruth.
"""

import argparse
import torch
import random
import numpy as np
import time
import matplotlib.pyplot as plt
import os.path as osp
import importlib

from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler 
import xml.etree.ElementTree as ET

from data.dataset import Pair_Dataset
from loss import criterion
import os.path as osp
from PIL import ImageFont
from utils import render_wo_gd

tensor2PIL = lambda x: transforms.ToPILImage()(x.view(-1, 512, 512))

def evaluate(args):
    ### DATA ###
    dataclass = Pair_Dataset(args.test_dir, test=True)
    imkey_list = dataclass.imkey_list
    dataloader = {}
    dataloader["test"] = DataLoader(dataclass,
                                  1, 
                                  shuffle=False, 
                                  num_workers=args.num_workers)
    fa, fb = open(args.deta_fn, "w"), open(args.detb_fn, "w")
    ### LOAD MODEL ###
    if osp.exists(args.model_fn):
        model_weights = torch.load(args.model_fn)
        model = importlib.import_module("model." + args.model).DiffNetwork()
        model.load_state_dict(model_weights)
        if args.cuda:
            model = model.cuda()
    else:
        raise IOError

    ### START TO EUALUATE ###
    tic = time.time()
    running_loss = 0

    for ii, (index, im_a, im_b, label) in enumerate(dataloader["test"]):
        inp_a, inp_b = Variable(im_a), Variable(im_b)
        if args.cuda:
            inp_a, inp_b = inp_a.cuda(), inp_b.cuda()
        pred = model(inp_a, inp_b)
        imkey = imkey_list[index[0]]
        
        deta_str, detb_str = render_wo_gd(args, imkey, pred)
        fa.write(deta_str), fb.write(detb_str)
        
    fa.close(), fb.close()

    print (" | Time consuming: {:.2f}s".format(time.time()-tic))

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, default="./data/test")
    parser.add_argument('--num_workers', type=int, default=8,
                            help="Number of data loading threads.")
    parser.add_argument('--no_cuda', action='store_true', default=False,
                            help="Disable CUDA training.")
    parser.add_argument('--model', type=str, default="base", 
                            help="A model name to generate network.")
    parser.add_argument('--model_fn', type=str, default="./default.pth.tar",
                            help="A model tar file to test.")
    parser.add_argument('--deta_fn', type=str, default="./result/det_a.txt", 
                            help="Detection result filename of image a.")
    parser.add_argument('--detb_fn', type=str, default="./result/det_b.txt", 
                            help="Detection result filename of image b.")
    parser.add_argument('--desdir', type=str, default="./result",
                            help="Rendered image directory.")
    parser.add_argument('--fontfn', type=str, default="./srcs/droid-sans-mono.ttf",
                            help="Font filename when rendering.")
    parser.add_argument('--render', type=int, default=0,
                            help="Output rendered files to result directory")

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args

if __name__ == "__main__":
    args = parse()
    evaluate(args)
    #u_test(args)
