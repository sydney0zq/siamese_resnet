#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017-10-25 21:09 zq <zq@mclab>
#
# Distributed under terms of the MIT license.

"""
Training siamese network.
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

def train(args):
    ### DATA ###
    dataloader = {}
    dataloader["train"] = DataLoader(Pair_Dataset(args.trainval_dir, train=True),
                                  args.batch_size, 
                                  shuffle=True, 
                                  num_workers=args.num_workers)
    dataloader["valid"] = DataLoader(Pair_Dataset(args.trainval_dir, train=False),
                                args.batch_size, 
                                shuffle=False, 
                                num_workers=args.num_workers)

    ### MODEL and METHOD ###
    model = SiameseBranchNetwork()
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.SGD(model.parameters(),
                                lr = args.lr,
                                momentum=0.9,
                                weight_decay = args.weight_decay)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_stepsize, gamma=0.1)
    #criterion = YOLO_loss()
    ### START TO MACHINE LEARNING ###
    tic = time.time()
    best_model_wts = model.state_dict()
    best_loss = float('inf')
    for epoch in range(args.nepochs):
        """
        Here I use the diff of image pair, that's because I think it may be a better feature.
        """
        print (" | Epoch {}/{}".format(epoch, args.nepochs-1))
        print (" | " + "-" * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                exp_lr_scheduler.step()
                model.train(True)   # Set model in training mode
            else:
                model.train(False)

            running_loss = 0
            for ii, (diff_ab, diff_ba, labela, labelb) in enumerate(dataloader[phase]):
                inp_ab, inp_ba = Variable(diff_ab), Variable(diff_ba)
                labela, labelb = Variable(labela), Variable(labelb)
                if args.cuda:
                    inp_ab, inp_ba = inp_ab.cuda(), inp_ba.cuda()
                    labela, labelb = labela.cuda(), labelb.cuda()
                optimizer.zero_grad()
                pred_ab, pred_ba = model(inp_ab, inp_ba)
                
                loss = criterion(labela, labelb, pred_ab, pred_ba)
                if phase == "train":
                    loss.backward()
                    optimizer.step()

                if ii % args.log_freq == 0 and phase == "train":
                    print (" | Epoch {}, Loss {:.2f}".format(epoch, loss.data[0]))
                running_loss += loss.data[0]
            epoch_loss = running_loss / (ii+1)
            print (" | {} Epoch Loss {:.2f}".format(phase, epoch_loss)) 

            # Deep copy of the model
            if phase == 'valid' and best_loss >= epoch_loss:
                best_loss = epoch_loss
                best_model_wts = model.state_dict()
                torch.save(best_model_wts, "./model_best.pth.tar")
                print (" | Epoch {} state saved, now loss reaches {}...".format(epoch, best_loss))
        print (" | Time consuming: {:.2f}s".format(time.time()-tic))
        print (" | ")
   


def parse():
    parser = argparse.ArgumentParser()
    ### DATA ###
    parser.add_argument('--trainval_dir', type=str, default="./data/train")
    parser.add_argument('--test_dir', type=str, default="./data/test")
    parser.add_argument('--nepochs', type=int, default=100,
                            help="Number of sweeps over the dataset to train.")
    parser.add_argument('--batch_size', type=int, default=4,
                            help="Number of images in each mini-batch.")
    parser.add_argument('--num_workers', type=int, default=8,
                            help="Number of data loading threads.")
    parser.add_argument('--no_cuda', action='store_true', default=False,
                            help="Disable CUDA training.")
    parser.add_argument('--model', type=str, default="", 
                            help="Give a model to test.")
    parser.add_argument('--lr', type=float, default=0.01, 
                            help="Learning rate for optimizing method.")
    parser.add_argument('--lr_stepsize', type=int, default=30, 
                            help="When val_loss increases, lr *= lr_decay.")
    parser.add_argument('--log_freq', type=int, default=5,)
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                            help="Goven the regularization term of the neural net.")

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    #print('Args: {}'.format(args))
    return args


if __name__ == "__main__":
    args = parse()
    train(args)




