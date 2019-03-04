# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 09:33:27 2018

@author: Morgan.Li
"""
from utility.math_operation import accuracy_process
from math import floor


default_focal = 918
camera_list = ['camera1', 'camera2', 'camera3', 'camera4', \
               'camera5', 'camera6', 'camera7', 'camera8']
len_relation = {"Red_Blue":0.4, "Red_Green": 0.6, "Blue_Green":1.0}
#center coordinate of image
"""
focal_scale is used to enlarge the focal length
default pixels: 1366*0.25 X 706*0.3 about 7million
mutiple focal_scale will enlarge the pixels of camera so that 
focal length will be changed
"""
focal_scale = 5

center_pt = (floor(focal_scale*1366*0.25*0.5), floor(focal_scale*706*0.3*0.5))

u0, v0 = center_pt


init_f = 1
scale_hw = 1  #scale of height and wide 
threshold = 0.0001







