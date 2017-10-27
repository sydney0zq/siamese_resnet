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

from model import SiameseBranchNetwork
from data.dataset import Pair_Dataset
from yolo_loss import criterion
import os.path as osp

from utils import getimsize, detrender, labelrender, parse_det

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
        # Evaluation
        inp_ab, inp_ba = Variable(diff_ab), Variable(diff_ba)
        labela, labelb = Variable(labela), Variable(labelb)
        if args.cuda:
            inp_ab, inp_ba = inp_ab.cuda(), inp_ba.cuda()
            labela, labelb = labela.cuda(), labelb.cuda()
        pred_ab, pred_ba = model(inp_ab, inp_ba)
        loss = criterion(labela, labelb, pred_ab, pred_ba)
        running_loss += loss.data[0]
        print (" | Eval {} Loss {:.2f}".format(ii+1, loss.data[0])) 
        print (" | Now start to generate detection files and render results...")
        
        imkey = int(imkey_list[index[0]])
        imsize = getimsize(args.test_dir, imkey)
        deta_str, deta_crd, gda_crd = parse_det(labela, pred_ab, imkey, imsize)
        detb_str, detb_crd, gdb_crd = parse_det(labelb, pred_ba, imkey, imsize)

        # Write str to det files
        fa.write(deta_str), fb.write(detb_str)
        
        # Render predictions
        detrender(args.test_dir, imkey, deta_crd, detb_crd, args.resdir)
        labelrender(args.resdir, imkey, gda_crd, gdb_crd)

    fa.close()
    fb.close()
    print (" | -- Eval Ave Loss {:.2f}".format(running_loss/(ii+1))) 
    print (" | Time consuming: {:.2f}s".format(time.time()-tic))

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
    parser.add_argument('--resdir', type=str, default="./result",
                            help="Rendered image directory.")

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    #print('Args: {}'.format(args))
    return args


if __name__ == "__main__":
    args = parse()
    evaluate(args)




