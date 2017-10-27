#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017-10-25 21:10 zq <zq@mclab>
#
# Distributed under terms of the MIT license.

"""
Dataset preprocessing class for dataloader.
"""
from torch.utils import data
import torch
import os
from PIL import Image
from PIL import ImageChops
from random import shuffle
import os.path as osp
from torchvision import transforms as T
from torch.utils.data import DataLoader
import numpy as np
import xml.etree.ElementTree as ET

def constrain(val, min_val, max_val):
    return min(max_val, max(min_val, val))


class Pair_Dataset(data.Dataset):

    def __init__(self, im_root, scale_size=512, label_shape=(5, 7, 7), transforms=None, train=True, test=False):
        """Get all images and spearate dataset to training and testing set."""
        self.test = test
        self.train = train
        self.im_root = im_root
        self.imkey_list = self.get_imkeylist()
        self.label_shape = label_shape
        self.scale_size = scale_size
        # Separate dataset
        if self.test:
            self.imkey_list = self.imkey_list
        elif self.train:
            self.imkey_list = self.imkey_list[:int(0.7*len(self.imkey_list))]
        else:
            self.imkey_list = self.imkey_list[int(0.7*len(self.imkey_list)):]
        self.imnum = len(self.imkey_list)

        # Transform
        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std =[0.229, 0.224, 0.225])
            # No enhancement on training and validating set
            self.transforms = T.Compose([T.Scale((scale_size, scale_size)), T.ToTensor(), normalize]) #T.RandomHorizontalFlip(),

    def __getitem__(self, index):
        """
        Return a pair of images and their corrsponding bounding box.
        """
        im_a_path = osp.join(self.im_root, "{:05d}".format(self.imkey_list[index])+"_a.jpg")
        im_b_path = osp.join(self.im_root, "{:05d}".format(self.imkey_list[index])+"_b.jpg")
        im_a, im_b = Image.open(im_a_path), Image.open(im_b_path)
        labela_path = osp.join(self.im_root, "{:05d}".format(self.imkey_list[index])+"_a.xml")
        labelb_path = osp.join(self.im_root, "{:05d}".format(self.imkey_list[index])+"_b.xml")
        diff_ab = self.transforms(ImageChops.subtract(im_a, im_b))
        diff_ba = self.transforms(ImageChops.subtract(im_b, im_a))

        labela, labelb = self.load_pair_label(labela_path, labelb_path, self.label_shape, self.scale_size)
        return index, diff_ab, diff_ba, labela, labelb
    
    def load_pair_label(self, labela_path, labelb_path, label_shape, scale_size):   # self unused
        """
        Return normalized groundtruth bboxes space.
        """
        labela = self.get_label(labela_path, label_shape, scale_size)
        labelb = self.get_label(labelb_path, label_shape, scale_size)
        return labela, labelb

    def get_label(self, label_path, label_shape, scale_size):
        label = np.zeros(label_shape)
        if osp.exists(label_path):
            tree = ET.parse(label_path)
            im_size = tree.findall("size")[0]
            ow = int(im_size.find("width").text)
            oh = int(im_size.find("height").text)
            sx = float(scale_size) / ow
            sy = float(scale_size) / oh
            boxes = []
            for obj in tree.findall('object'):
                bbox = obj.find('bndbox')
                t_boxes = [int(bbox.find('xmin').text), int(bbox.find('ymin').text),
                           int(bbox.find('xmax').text), int(bbox.find('ymax').text)] 
                boxes.append([1,
                            (t_boxes[0] + t_boxes[2])/(2.0 * ow), # norm center x
                            (t_boxes[1] + t_boxes[3])/(2.0 * oh), # norm center y
                            (t_boxes[2] - t_boxes[0])*1.0/ow,     # norm w
                            (t_boxes[3] - t_boxes[1])*1.0/oh])    # norm h
            # scale and correct boxes(cuz we resize input images to scale_size*scale_size)
            for i in range(len(boxes)):
                left = (boxes[i][1] - boxes[i][3] / 2) * sx       # left mid point
                right = (boxes[i][1] + boxes[i][3] / 2) * sx      # right mid point
                top = (boxes[i][2] - boxes[i][4] / 2) * sy        # top mid point
                bottom = (boxes[i][2] + boxes[i][4] / 2) * sy     # bottom mid point
                left = constrain(0, 1, left)
                right = constrain(0, 1, right)
                top = constrain(0, 1, top)
                bottom = constrain(0, 1, bottom)
                boxes[i][1] = (left + right) / 2
                boxes[i][2] = (top + bottom) / 2
                boxes[i][3] = right - left
                boxes[i][4] = bottom - top
            # In python3 range is a generator object - it does not return a list. Convert it to a list before shuffling.
            lst = list(range(len(boxes)))       
            shuffle(lst)
            for i in lst:
                x, y, w, h = boxes[i][1:]
                if (w < 0.01 or h < 0.01):
                    continue
                col = int(x * label_shape[2])
                row = int(y * label_shape[1])
                x = x * label_shape[2] - col
                y = y * label_shape[1] - row
                if label[0, row, col] != 0:
                    continue
                label[:, row, col] = 1, x, y, w, h
        return label

    def get_imkeylist(self):
        imkey_list = []
        for i in os.listdir(self.im_root):
            if i[-3:] == "jpg" and (int(i[:-6]) not in imkey_list) and ("diff" not in i):
                imkey_list.append(int(i[:-6]))
        return imkey_list
    
    def __len__(self):
        """
        Return image pair number of the dataset.
        """
        return self.imnum


if __name__ == "__main__":
    train_dataset = Pair_Dataset("/home/zq/diffproj/data/train", train=True)
    trainloader = DataLoader(train_dataset, 
                             batch_size=4,
                             shuffle=True,
                             num_workers=4)
    #print (np.sort(train_dataset.imkey_list))
    #print (len(train_dataset.imkey_list))
    #exit()
    #for ii, (im, label) in enumerate(trainloader):
    for ii, (im_a, im_b, labela, labelb) in enumerate(trainloader):
        print (ii, im_a.size(), labela.shape, im_b.size(), labelb.shape)
        print (type(im_a), type(labela))
        print (labela.shape[2]*labela.shape[3])
        exit()

