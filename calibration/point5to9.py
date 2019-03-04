# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 11:19:18 2018

@author: Morgan.Li
"""
import numpy as np
from numpy import sin, cos

def point5to9(X1,length_known):
    """
    POINT5TO9 gives the coordinates of collinear points A,B,C
    
    Input   - X1             -> (5xM)  coordinates of 3D points
            - length_known   -> (1x2)  actural length of feature points on the calibration wand
    
    Output  - X2             -> (3xN)  coordinates of 3D points
    """
    
    M = X1.shape[-1]
    
    X2 = np.zeros((3, 3*M))    

    for i in np.arange(M):
        direct_scale = np.array([ \
          sin(X1[3,i])*cos(X1[4,i]), sin(X1[3,i])*sin(X1[4,i]), cos(X1[3,i])])
   
        X2[:,3*i]=X1[0:3,i]
             
        X2[:,3*i + 1]=X1[0:3,i]+ length_known[0]*direct_scale

        X2[:,3*i + 2]=X1[0:3,i] + length_known[1]*direct_scale
    
    return X2





