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

from model import SiameseBranchNetwork
from data.dataset import Pair_Dataset
from yolo_loss import criterion
import os.path as osp

def evaluate(args):
    ### DATA ###
    dataloader = {}
    dataloader["test"] = DataLoader(Pair_Dataset(args.test_dir, train=True),
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

    for ii, (diff_ab, diff_ba, labela, labelb) in enumerate(dataloader["test"]):
        inp_ab, inp_ba = Variable(diff_ab), Variable(diff_ba)
        labela, labelb = Variable(labela), Variable(labelb)
        if args.cuda:
            inp_ab, inp_ba = inp_ab.cuda(), inp_ba.cuda()
            labela, labelb = labela.cuda(), labelb.cuda()
        pred_ab, pred_ba = model(inp_ab, inp_ba)
        
        loss = criterion(labela, labelb, pred_ab, pred_ba)
        print (" | Eval Loss {:.2f}".format(loss.data[0])) 

    print (" | Time consuming: {:.2f}s".format(time.time()-tic))
    print (" | ")
   


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

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    #print('Args: {}'.format(args))
    return args


if __name__ == "__main__":
    args = parse()
    evaluate(args)




