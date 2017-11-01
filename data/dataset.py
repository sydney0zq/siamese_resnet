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
import random
import xml.etree.ElementTree as ET


### UTEST ###
#from u_test import labelrender

def constrain(min_val, max_val, val):
    return min(max_val, max(min_val, val))

class Pair_Dataset(data.Dataset):

    """
        label_shape: 10 means __(00, 01, 10, 11)________(two boundingboxes)
    """
    def __init__(self, im_root, scale_size=512, label_shape=(7, 7, 7), transforms=None, train=True, test=False):
        """Get all images and spearate dataset to training and testing set."""
        self.test, self.train = test, train
        self.im_root = im_root
        self.imkey_list = self.get_imkeylist()
        self.label_shape = label_shape
        self.scale_size = scale_size
        self.tensor2PIL = T.Compose([T.ToPILImage()])
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
            # 在ToTensor这个类中已经实现了0~1的映射
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std =[0.229, 0.224, 0.225])
            # No enhancement on training and validating set
            self.transforms = T.Compose([T.Scale((scale_size, scale_size)), 
                                         T.CenterCrop((scale_size, scale_size)),
                                         T.ToTensor(), normalize]) #T.RandomHorizontalFlip(),
                                         #T.ToTensor()]) #T.RandomHorizontalFlip(),

    def __getitem__(self, index):
        """ Return a pair of images and their corrsponding bounding box. """
        im_a_path = osp.join(self.im_root, "{:05d}".format(self.imkey_list[index])+"_a.jpg")
        im_b_path = osp.join(self.im_root, "{:05d}".format(self.imkey_list[index])+"_b.jpg")
        im_a, im_b = Image.open(im_a_path), Image.open(im_b_path)
        im_a = self.transforms(im_a)
        im_b = self.transforms(im_b)
        if random.uniform(0, 1) > 0.5:
            im_a, im_b = im_b, im_a

        """ Return normalized groundtruth bboxes space. """
        labela_path = osp.join(self.im_root, "{:05d}".format(self.imkey_list[index])+"_a.xml")
        labelb_path = osp.join(self.im_root, "{:05d}".format(self.imkey_list[index])+"_b.xml")
        #labela_path = "/home/zq/diff_resnet/data/test/00590_a.xml"
        #labelb_path = "/home/zq/diff_resnet/data/test/00590_b.xml"
        label = self.load_pair_label(labela_path, labelb_path)
        return index, im_a, im_b, label

    def load_pair_label(self, labela_path, labelb_path):
        labela = self.get_label(labela_path)
        labelb = self.get_label(labelb_path)
        return self.mergelabel(labela, labelb)

    ###################################################
    # GET LABEL TO STANDARD FORMAT
    ###################################################
    def get_label(self, label_path):
        ROW, COL = self.label_shape[1], self.label_shape[2]
        label = np.zeros((5, 7, 7)) # FIXED
        if osp.exists(label_path):
            tree = ET.parse(label_path)
            im_size = tree.findall("size")[0]
            ow, oh = int(im_size.find("width").text), int(im_size.find("height").text)
            bboxes = []
            for obj in tree.findall('object'):
                bbox = obj.find('bndbox')
                t_boxes = [int(bbox.find('xmin').text), int(bbox.find('ymin').text),
                           int(bbox.find('xmax').text), int(bbox.find('ymax').text)] 
                bboxes.append([1,
                            (t_boxes[0] + t_boxes[2])/(2.0*ow), # center x
                            (t_boxes[1] + t_boxes[3])/(2.0*oh), # center y
                            (t_boxes[2] - t_boxes[0])*1.0/ow,  # w
                            (t_boxes[3] - t_boxes[1])*1.0/oh]) # h
            #print ("*"* 30)
            #print (bboxes)

            lst = list(range(len(bboxes)))       
            shuffle(lst)
            for i in lst:
                x, y, w, h = bboxes[i][1:]
                x, y, w, h = constrain(0, 1, x), constrain(0, 1, y), constrain(0, 1, w), constrain(0, 1, h)
                if (w < 0.01 or h < 0.01):
                    continue
                col, row = int(x * self.label_shape[2]), int(y * self.label_shape[1])
                x, y = x * self.label_shape[2] - col, y * self.label_shape[1] - row
                if label[0, row, col] != 0:
                    continue
                label[0, row, col] = 1
                label[1:, row, col] = x, y, w, h
        return label
    
    def mergelabel(self, labela, labelb):
        label = np.zeros(self.label_shape)
        for row in range(self.label_shape[1]):
            for col in range(self.label_shape[2]):
                if labela[0, row, col] == 1 and label[1, row, col] == 0:
                    label[0, row, col] = 1
                    label[1, row, col] = 1
                    label[3:7, row, col] = labela[1:, row, col]
                if labelb[0, row, col] == 1 and label[2, row, col] == 0 and label[1, row ,col] == 0:
                    label[0, row, col] = 1
                    label[2, row, col] = 1
                    label[3:7, row, col] = labelb[1:, row, col]
        return label

    def get_imkeylist(self):
        imkey_list = []
        for i in os.listdir(self.im_root):
            if i[-3:] == "jpg" and (int(i[:-6]) not in imkey_list) and ("diff" not in i):
                imkey_list.append(int(i[:-6]))
        return imkey_list
    
    def __len__(self):
        """ Return image pair number of the dataset. """
        return self.imnum


if __name__ == "__main__":
    train_dataset = Pair_Dataset("./test", test=True)
    trainloader = DataLoader(train_dataset, 
                             batch_size=1,
                             shuffle=True,
                             num_workers=0)
    #print (np.sort(train_dataset.imkey_list))
    #print (len(train_dataset.imkey_list))
    #for ii, (im, label) in enumerate(trainloader):
    imkeys = train_dataset.imkey_list
    for ii, (index, im_a, im_b, label) in enumerate(trainloader):
        #print (ii, im_a.size(), labela.shape, im_b.size(), labelb.shape)
        #print (type(im_a), type(labela))
        #print (labela.shape[2]*labela.shape[3])
        #print (index[:], "-----", ii)
        #print (imkeys[index[0]])
        #exit()
        print (label)
        exit()
        pass

        """
        diff_ab = self.transforms(ImageChops.subtract(im_a, im_b))
        diff_ba = self.transforms(ImageChops.subtract(im_b, im_a))
        ### TEST ###
        dab = diff_ab.numpy()
        dba = diff_ba.numpy()
        #dab[dab < 0.5] = 0
        dab = self.tensor2PIL(torch.from_numpy(dab))
        #draw_bbox(dab, )
        dab.save("/tmp/test/" + "{:05d}".format(index) + "_render_a.jpg")
        t = ImageChops.subtract(im_a, im_b)
        t.save("/tmp/ori/" + "{:05d}".format(index) + "_render_a.jpg")
        #dba[dba < 0.5] = 0
        dba = self.tensor2PIL(torch.from_numpy(dba))
        dba.save("/tmp/test/" + "{:05d}".format(index) + "_render_b.jpg")
        t = ImageChops.subtract(im_b, im_a)
        t.save("/tmp/ori/" + "{:05d}".format(index) + "_render_b.jpg")
        posa = np.where(labela[0] == 1)
        posb = np.where(labelb[0] == 1)
        bboxa = labela[1:, posa[0], posa[1]].transpose()
        bboxb = labelb[1:, posb[0], posb[1]].transpose()
        def r(bbox, im_size):
            bbox[:, 0], bbox[:, 2] = bbox[:, 0]*im_size[0], bbox[:, 2]*im_size[0]
            bbox[:, 1], bbox[:, 3] = bbox[:, 1]*im_size[1], bbox[:, 3]*im_size[1]
            return bbox
        bboxa = r(bboxa, im_a.size)
        bboxb = r(bboxb, im_a.size)
        labelrender("/tmp/ori", index, bboxa, bboxb)
        return index, im_a, im_b, labela, labelb
        """
