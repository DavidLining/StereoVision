# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 11:18:29 2018

@author: Morgan.Li
"""

from calibration.five_point_algorithm import five_point_algorithm
import numpy as np
import random
from numpy.linalg import inv
from utility.error import InvalidDenomError
# ransac_five.m

    # RANSAC_FIVE - RANSAC algorithm for the estimation of the essential
# matrix using the five point algorithm
    
    
    # This is a ransac implementation using the calibrated five point solver of
# Henrik Stewenius. Due to license constraints we are unable to distribute
# his code and refer the user to http://vis.uky.edu/~stewe. Please read the
# INSTALLATION.txt before attempting to use this.
    
    
    # Input  - X1           -> (3xn) set of spherical points corresponding to image points in image 1
#        - X2           -> (3xn) set of spherical points corresponding to image points in image 2
#        - N            -> (1x1) Number of iterations
#        - threshold    -> (1x1) Threshold to use
    
    # Output - bestE        -> (3x3) Essential matrix E
#        - bestmodel    -> (nx1) Set of inliers
    
    
    
    # Author: Isaac Esteban
# IAS, University of Amsterdam
# TNO Defense, Security and Safety
# isaac@fit3d.info
# isaac.esteban@tno.nl
    
def ransac_five(X1, X2, K1, K2, N = 1000, threshold = 0.001,*args,**kwargs):

    # Parameters
    t=threshold
    # Best scores
    bestscore=0
    bestmodel=np.zeros((3,3))

    bestE=np.zeros((3,3))

    # Normalization for 5-point
    X1n=X1
    X2n=X2

    counter=1
    while (counter < N):
        rindex = random.sample(range(X1.shape[1]), 5)
        E5=five_point_algorithm(X1n[:,rindex],X2n[:,rindex], K1, K2)
        counter=counter + 1
        for Et in E5:
            Ft = inv(K2.T).dot(Et).dot(inv(K1))
            X2tFX1 = []
            for idx in range(X1.shape[1]):
                X2tFX1.append(X2n[:,idx].T.dot(Ft).dot(X1n[:,idx]))
            X2tFX1 = np.array(X2tFX1)    
#            FX1 = Ft.dot(X1n)
#            FtX2 = Ft.T.dot(X2n)
            #d is a matrix 1*size(X1, 2)
            #d=X2tFX1 ** 2 / (FX1[0,:] ** 2 + FX1[1,:] ** 2 + FtX2[0,:] ** 2 + FtX2[1,:] ** 2)           
            residual_error = X2tFX1
            inliersL= [i for i in range(len(residual_error)) if abs(residual_error[i]) < t]
            
            #largest number of point than lower than limit 't' -Morgan
            if (len(inliersL) > bestscore and len(inliersL) > 4):
                bestscore=len(inliersL)
                bestmodel=inliersL
                bestE=Et
            
    #print(residual_error)
    return bestmodel, bestE       
    #     # Reestimate F based on the inliers of the best model only
'''
    if(len(X1[:,bestmodel]) >= 5):
        E5=five_point_algorithm(X2n[:,bestmodel],X1n[:,bestmodel], K1, K2)
        for Et in E5:
            X2tFX1 = []
            for idx in range(X1.shape[1]):
                X2tFX1.append(X2n[:,idx].T.dot(Et).dot(X1n[:,idx]))
                
            FX1 = Et.dot(X1n)
            FtX2 = Et.T.dot(X2n)
                
            #d is a matrix 1*size(X1, 2)
            d=X2tFX1 ** 2 / (FX1[1,:] ** 2 + FX1[2,:] ** 2 + FtX2[1,:] ** 2 + FtX2[2,:] ** 2)
            
            inliersL= len([i for i in range(len(d)) if abs(d[i]) < t])
    
            #largest number of point than lower than limit 't' -Morgan
            if (len(inliersL) > bestscore and len(inliersL) > 4):
                bestscore=len(inliersL)
                bestmodel=inliersL
                bestE=Et
'''    



#Test code
#K1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
#K2 = K1
#pts1 = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [1, 1, 1, 1, 1]])
#pts2 = np.array([[3, 3, 4, 5, 6], [8, 9, 7, 6, 8], [1, 1, 1, 1, 1]])
#bestE = ransac_five(pts1, pts2, K1, K2, 1000, 0.001)
#print(bestE)

