# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 14:19:53 2018

@author: Morgan.Li
"""
from math import floor
import numpy as np
    
def accuracy_process(value, keep_reso=False):
    if keep_reso:
        return (floor(value) + 0.5)
    else:
        return (value)        

def get_rotate_matrix(three_points_coor1, three_points_coor2):
    """
    three_points_coor1: coordinates of three points before rotate
    three_points_coor2: coordinates of three points after rotate
    """
    rotate_m = np.dot(np.matrix(three_points_coor1).I, np.matrix(three_points_coor2))
    return rotate_m
   