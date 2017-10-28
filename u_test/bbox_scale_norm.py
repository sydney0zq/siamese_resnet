#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017-10-28 01:43 zq <zq@mclab>
#
# Distributed under terms of the MIT license.

"""
A scipt to fuck off the origin code.
"""
import numpy as np
# 577
sx,sy = 960/512.0, 540/512.0
#sx,sy = 720/512.0, 540/512.0
#b = np.array([171, 72, 330, 171])
#b = np.array([387, 347, 436, 401])
b = np.array([244, 51, 392, 182])
print (b)
xmin, ymin, xmax, ymax = b[:]
mid = np.array([(xmin+xmax)/2.0, (ymin+ymax)/2.0, (xmax-xmin), (ymax-ymin)])
t = np.array([mid[0]/sx, mid[1]/sy, mid[2]/sx, mid[3]/sy])
t = t/512.
print (t)
#t =np.array( [0.97, 0.49, 0.08, 0.173])
#t =np.array([0.41736111 ,0.96481481 , 0.09305556 , 0.21851852])
#t = np.array([0.79027778,  0.16574074  ,0.08611111,  0.12407407])
t = np.array([   6.94444444e-04 ,  8.48148148e-01 ,  6.80555556e-02,0.1 ])
tu = t*512.
t2 = np.array([tu[0]*sx, tu[1]*sy, tu[2]*sx, tu[3]*sy])

midx, midy, w, h = t2[:]
t3 = [midx-w/2., midy-h/2., midx+w/2., midy+h/2.]
print (t3)




