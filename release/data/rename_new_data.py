#!/usr/bin/env python3
import os
from itertools import zip_longest
import os.path as osp

"""
Rename corrsponding image and labels in `DIR` to specified format.
"""

# DIR of image root
DIR="00"
# The beginning number of images, you should notice it is not existed in original dataset
BASE=1000

flist = os.listdir(DIR)
jlist = []
## Generate jpg lists
for fn in flist:
    if fn.endswith(".jpg"):
        jlist.append(fn)

for i in range(0, len(jlist), 2):
    im_a, im_b = jlist[i:i+2]
    nim_a, nim_b = "{:04d}".format(BASE+i) + "_a.jpg", "{:04d}".format(BASE+i) + "_b.jpg"
    cmd1 = "mv " + osp.join(DIR, im_a) + " " + osp.join(DIR, nim_a)
    cmd2 = "mv " + osp.join(DIR, im_b) + " " + osp.join(DIR, nim_b)
    os.system(cmd1), os.system(cmd2)
    print (cmd1, cmd2)
    xml_a, xml_b = im_a.replace("jpg", "xml"), im_b.replace("jpg", "xml")
    nxml_a, nxml_b = "{:04d}".format(BASE+i) + "_a.xml", "{:04d}".format(BASE+i) + "_b.xml"
    xcmd1 = "mv " + osp.join(DIR, xml_a) + " " + osp.join(DIR, nxml_a)
    xcmd2 = "mv " + osp.join(DIR, xml_b) + " " + osp.join(DIR, nxml_b)
    if osp.exists(osp.join(DIR, xml_a)):
        os.system(xcmd1)
        print(xcmd1)
    if osp.exists(osp.join(DIR, xml_b)):
        os.system(xcmd2)
        print(xcmd2)





