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

def constrain(min_val, max_val, val):
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
            # NOTE 我认为这个归一化的数字对于这个任务假如使用差值图像输入没有任何意义
            # 并且在ToTensor这个类中已经实现了0~1的映射
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std =[0.229, 0.224, 0.225])
            # No enhancement on training and validating set
            self.transforms = T.Compose([T.Scale((scale_size, scale_size)), 
                                         T.CenterCrop((scale_size, scale_size)),
                                         T.ToTensor()]) #T.RandomHorizontalFlip(),
                                         #T.ToTensor(), normalize]) #T.RandomHorizontalFlip(),

    def __getitem__(self, index):
        """
        Return a pair of images and their corrsponding bounding box.
        """
        im_a_path = osp.join(self.im_root, "{:05d}".format(self.imkey_list[index])+"_a.jpg")
        im_b_path = osp.join(self.im_root, "{:05d}".format(self.imkey_list[index])+"_b.jpg")
        #im_a_path = "./test/00553_a.jpg"
        #im_b_path = "./test/00553_b.jpg"
        im_a, im_b = Image.open(im_a_path), Image.open(im_b_path)

        labela_path = osp.join(self.im_root, "{:05d}".format(self.imkey_list[index])+"_a.xml")
        labelb_path = osp.join(self.im_root, "{:05d}".format(self.imkey_list[index])+"_b.xml")
        labela, labelb = self.load_pair_label(labela_path, labelb_path, self.label_shape, self.scale_size)

        #diff_ab = self.transforms(im_a)
        #diff_ba = self.transforms(im_b)
        
        diff_ab = self.transforms(ImageChops.subtract(im_a, im_b))
        diff_ba = self.transforms(ImageChops.subtract(im_b, im_a))
        
        """
        dab = diff_ab.numpy()
        print (dab)
        dba = diff_ba.numpy()
        dab[dab < 0.5] = 0
        dab = self.tensor2PIL(torch.from_numpy(dab))
        dba[dba < 0.5] = 0
        dba = self.tensor2PIL(torch.from_numpy(dba))

        im_a = self.transforms(im_a)
        print ("im_a.tensor.shape", im_a.size)
        im_a = im_a.numpy()
        print ("im_a.shape", im_a.shape)
        im_a =  self.tensor2PIL(torch.from_numpy(im_a))

        im_a.save("a.png")
        im_b.save("b.png")
        dab.save("dab.png")
        dba.save("dba.png")
        exit()
        """
        return index, diff_ab, diff_ba, labela, labelb
        
    def load_pair_label(self, labela_path, labelb_path, label_shape, scale_size):
        """ Return normalized groundtruth bboxes space. """
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
            #print ("ow oh", ow, oh)
            sx = ow / float(scale_size)
            sy = oh / float(scale_size)
            bboxes = []
            for obj in tree.findall('object'):
                bbox = obj.find('bndbox')
                t_boxes = [int(bbox.find('xmin').text), int(bbox.find('ymin').text),
                           int(bbox.find('xmax').text), int(bbox.find('ymax').text)] 
                #print ("gd xmin, ymin, xmax, ymax", t_boxes)
                bboxes.append([1,
                            (t_boxes[0] + t_boxes[2])/2.0, # center x
                            (t_boxes[1] + t_boxes[3])/2.0, # center y
                            (t_boxes[2] - t_boxes[0])*1.0,  # w
                            (t_boxes[3] - t_boxes[1])*1.0]) # h
            # scale and correct boxes(cuz we resize input images to scale_size*scale_size)
            for i in range(len(bboxes)):
                midx, midy, w, h = bboxes[i][1:]
                scale_bbox = [midx/sx, midy/sy, w/sx, h/sy]
                norm_bbox = [x*1./scale_size for x in scale_bbox]
                bboxes[i][1:] = norm_bbox
                #print ("norm_bbox", norm_bbox)
                
            # In python3 range is a generator object - it does not return a list. Convert it to a list before shuffling.
            lst = list(range(len(bboxes)))       
            shuffle(lst)
            for i in lst:
                x, y, w, h = bboxes[i][1:]
                x = constrain(0, 1, x)
                y = constrain(0, 1, y)
                w = constrain(0, 1, w)
                h = constrain(0, 1, h)
                if (w < 0.01 or h < 0.01):
                    continue
                col = int(x * label_shape[2])
                row = int(y * label_shape[1])
                if label[0, row, col] != 0:
                    continue
                label[:, row, col] = 1, x, y, w, h
                #print ("label", label[:, row, col])
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
    train_dataset = Pair_Dataset("./test", test=True)
    trainloader = DataLoader(train_dataset, 
                             batch_size=1,
                             shuffle=True,
                             num_workers=0)
    #print (np.sort(train_dataset.imkey_list))
    #print (len(train_dataset.imkey_list))
    #exit()
    #for ii, (im, label) in enumerate(trainloader):
    imkeys = train_dataset.imkey_list
    for ii, (index, im_a, im_b, labela, labelb) in enumerate(trainloader):
        #print (ii, im_a.size(), labela.shape, im_b.size(), labelb.shape)
        #print (type(im_a), type(labela))
        #print (labela.shape[2]*labela.shape[3])
        #print (index[:], "-----", ii)
        print (imkeys[index[0]])
        exit()

