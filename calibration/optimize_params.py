# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 16:33:09 2018

@author: Morgan.Li
"""

"""
http://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html

"""
import numpy as np
from scipy.optimize import minimize, least_squares
from calibration.calibrate import StereoCalibrator
from cv2 import Rodrigues
from calibration.calib_config import len_relation, scale_hw
from calibration.internal_params import skew, triangulate_point, \
                     triangulate_point_opt, triangulate_pts_opt
import time
from numpy.linalg import inv

len_relation = len_relation    
scale_hw = scale_hw
        
def compute_2points_len_error(img_pts1, img_pts2, proj_M1, proj_M2, stand_lens):
    """
    compute the length error.
    1. get the coordinates of three points
    2. compute the distance between each two points and compare with true value
    
    stand_len: the standard length of two points
    
    """
    #red blue green
 
    n = img_pts1.shape[1]
    if img_pts2.shape[1] != n:
        raise ValueError("Number of points don't match.")
        
        
    error1_array, error2_array = triangulate_pts_opt(img_pts1, img_pts2, \
                                                     proj_M1, proj_M2, stand_lens, 27)

#    error1_array =  np.zeros((1, n))[0]
#    error2_array =  np.zeros((1, int(n/3)))[0]   
#    for i in np.arange(int(n/3)):
#
#        pt_Red = triangulate_point_opt(img_pts1[:,3*i],img_pts2[:,3*i],\
#                                   proj_M1,proj_M2)
#        pt_Blue = triangulate_point_opt(img_pts1[:,3*i+1],img_pts2[:,3*i+1],\
#                                    proj_M1,proj_M2)
#        pt_Green = triangulate_point_opt(img_pts1[:,3*i+2],img_pts2[:,3*i+2],\
#                                     proj_M1,proj_M2)
#        
#        pts_m = np.row_stack((pt_Red - pt_Green, \
#                              pt_Red - pt_Blue, \
#                              pt_Blue - pt_Green))
#        dist_V = np.linalg.norm(pts_m, axis=1)
##       dist_V = get_dist(pt_Red, pt_Blue, pt_Green)        
#
#        error1_array[3*i: 3*i+3]  =  dist_V - stand_lens
#        error2_array[i] = dist_V[2] - dist_V[1] - dist_V[0]
    
    return error1_array, error2_array


def get_dist(pt_Red, pt_Blue, pt_Green):
    R_B = pt_Red - pt_Blue
    R_G = pt_Red - pt_Green
    B_G = pt_Blue - pt_Green 
    return np.array([R_B.dot(R_B), R_G.dot(R_G), B_G.dot(B_G)])

def compute_points_image_error(img_pts1, img_pts2, K1, K2, R2, T2):
    """
    the residual error with respect to an estimated
    essential matrix E^ is given by: |a1.T*F*a0|
    
    
    """
    Et = skew(T2).dot(R2) #E = SR
    #fundamental matrix
    Ft = inv(K2.T).dot(Et).dot(inv(K1))
       

    n = img_pts1.shape[1]
    if img_pts2.shape[1] != n:
        raise ValueError("Number of points don't match.")
#    error_array =   np.zeros((1, n))[0]        
#    for i in np.arange(int(n/3)):
#
#        a2 = img_pts2[:,3*i:3*i+3] 
#        a1 = img_pts1[:,3*i:3*i+3]
#        error = a2.T.dot(Ft).dot(a1)
#        error_array[3*i:3*i+3] = np.diag(error)
        
    error_array  = np.einsum('ij,ji->i', img_pts2.T, Ft.dot(img_pts1))

    return error_array



def lose_func(x, img_pts0, img_pts1, len_relation, focals=None):
    """
    lost functions for optimize the params during the stereo calibration
    """
    if focals:
        f0, f1 = focals
        r1x, r1y, r1z, t1x, t1y, t1z = x
    else:
        f0, f1, r1x, r1y, r1z, t1x, t1y, t1z = x

    weight_dist_e = 40 #the weight of distance error
    weight_delta_e = 10
    weight_img_e = 1 #
    K0 = np.diag([f0, scale_hw*f0, 1.0])
    K1 = np.diag([f1, scale_hw*f1, 1.0])
    
    proj_M0 = np.column_stack((K0, [0, 0, 0]))
    R1 = np.array([r1x, r1y, r1z])
    R_M1 = Rodrigues(R1)[0] #vector to matrix
    #R_M1 = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])
    t_V1 = np.array([ t1x, t1y, t1z])
    R_t_M1 = np.column_stack((R_M1, t_V1))
    proj_M1 = K1.dot(R_t_M1)
    stand_lens = np.array([len_relation["Red_Green"],\
                           len_relation["Red_Blue"], \
                           len_relation["Blue_Green"]])
 
    g1, g2 = compute_2points_len_error(img_pts0, img_pts1, proj_M0, proj_M1, stand_lens)

    g3 = compute_points_image_error(img_pts0, img_pts1, K0, K1, R_M1, t_V1.tolist())

    
    return np.concatenate((weight_dist_e*g1, weight_delta_e*g2, weight_img_e*g3),axis=0)



def optimize_params_two_camera(init_params, img_pts0, img_pts1, len_relation, focals=None):
    """
     optimize the params during the stereo calibration
     Method   Loss       Test cost Time(optimize + bundle adjust)
     'lm'     'linear'   140+39s
     'trf'    'linear'   126+42s
     'trf'    'soft_l1'  108+38s
     'trf'    'huber'    103+38s 
    """
    t0 = time.time()
    img_pts0_3XN = img_pts0
    img_pts1_3XN = img_pts1    
#    img_pts0_9XM = img_pts0.T.reshape((-1, 9)).T
#    img_pts1_9XM = img_pts1.T.reshape((-1, 9)).T
    params = least_squares(lose_func, init_params, \
                           loss='huber', \
                           args = (img_pts0_3XN, img_pts1_3XN, len_relation, focals))
    
    if focals:
        op_x = list(focals) + params.x.tolist()
    else:
        op_x = params.x.tolist()
    t1 = time.time()
    print("Optimization take {0:.0f} seconds".format(t1 - t0))
    return op_x


