#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017-10-25 15:52 zq <zq@mclab>
#
# Distributed under terms of the MIT license.

"""

"""

import sys
import gflags
import os
import time
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from PIL import ImageChops
from PIL import ImageDraw
import numpy as np
import os.path as osp

FLAGS = gflags.FLAGS
gflags.DEFINE_string('detfn', "./det_a.txt",
        """Detection in PASCAL VOC format.""")
gflags.DEFINE_string('imdir', "../data/test",
        """Corrsponding image directory.""")
gflags.DEFINE_string('rendir', "./",
        """Corrsponding rendered image directory.""")

floor = lambda x: math.floor(float(x))


def draw_multibbox(im_fn, imkey, det_db, rendim_fn):
    im = Image.open(im_fn)
    draw_im = ImageDraw.Draw(im)

    for i_det in det_db:
        det_key, i_prob, xmid, ymid, w, h = [float(x) for x in i_det[0]]
        #print ("imkey", imkey, ":", "det_key", det_key)
        print (" I am drawing ", imkey)
        draw_im.line([(floor(xmid-w/2.0), floor(ymid-h/2.0)), (floor(xmid+w/2.0), floor(ymid-h/2.0))])
        #draw_im.line([floor(xmid-w/2.0), floor(ymid-h/2.0), floor(xmid-w/2.0), floor(ymid+h/2.0)])
        #draw_im.line([floor(xmid-w/2.0), floor(ymid+h/2.0), floor(xmid+w/2.0), floor(ymid+h/2.0)])
        #draw_im.line([floor(xmid+w/2.0), floor(ymid-h/2.0), floor(xmid+w/2.0), floor(ymid+h/2.0)])
    del draw_im
    im.save(rendim_fn)

def main(detfn, imdir, rendir):
    tic = time.time() 
    imkey_list = []
    with open(detfn, "r") as f:
        det = f.readlines()
        det = [x.strip() for x in det]
        for i_det in det:
            imkey_list.append(i_det[0:5])
    det_db = [[x.split(" ")] for x in det]
    print (" | Load {} in {} seconds...".format(detfn, time.time()-tic))

    tic = time.time() 
    print (" | Start to render images...")
    for imkey in imkey_list:
        draw_multibbox(os.path.join(imdir, imkey+"_a.jpg"), imkey, det_db, os.path.join(rendir, imkey+"_render_a.jpg"))
    #    draw_multibbox(os.path.join(imdir, imkey+"_b.jpg"), imkey, det_db, os.path.join(rendir, imkey+"_render_b.jpg"))

    print (" | {} in {} seconds...".format("Rendered", time.time()-tic))

if __name__ == "__main__":
    FLAGS(sys.argv)
    main(FLAGS.detfn, FLAGS.imdir, FLAGS.rendir)





