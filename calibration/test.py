# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 14:30:55 2018

@author: Morgan.Li
"""


import numpy as np
from calibration.point5to9 import point5to9
from calibration.point9to5 import point9to5

ref_coord_pts = np.array([[-0.1, -0.5, 0.5], \
                          [0.125, 0.125, 0.125],\
                          [ -0.25, -0.25, -0.25]])    



points5 = point9to5(ref_coord_pts)
points9 = point5to9(points5, [0.4, 0.6])
print(points5)
print(points9)