# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 11:19:41 2018

@author: Morgan.Li
"""

import numpy as np
from numpy import arccos, sign, pi, arctan
from math import floor

def point9to5(X1):
    """
    POINT9TO5 gives the direction represention of collinear points A,B,C
    
    Input  - X1          -> (3xN)  coordinates of 3D points
    
    Output - X2          -> (5xM)  coordinates of 3D points
    """
    
    M=floor(X1.shape[-1]/3)
    X2 = np.zeros((5, M))
    for i in np.arange(M):
        X2[0:3,i]=X1[0:3,i]
        point_direction=(X1[0:3,i+ 2*M] - X1[0:3,i]) / np.linalg.norm(X1[0:3,i+ 2*M] - X1[0:3,i])
        X2[3,i]= arccos(point_direction[-1])
        if point_direction[0] == 0:
            if point_direction[1] == 0:
                X2[4,i]=0
            else:
                X2[4,i]= sign(point_direction[1])*pi / 2
        else:
            X2[4,i]= arctan(point_direction[1] / point_direction[0])
        if point_direction[0] < 0:
            X2[4,i]=X2[4,i] + pi

        if X2[4,i] < 0:
            X2[4,i]=X2[4,i] + 2*pi

    return X2



