# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 16:32:50 2018

@author: Morgan.Li
"""
import numpy as np
from scipy import linalg
from calibration.calibrate import StereoCalibrator
from calibration.internal_params import triangulate_point
from numpy.linalg import inv
from scipy import optimize

def compute_coord_from_triangule_locate_eqs(p,pts_world_csys, pts_dist):
    """
    equations used for function 'compute_coord_from_triangule_locate'
    """
    x, y, z = p
    n = pts_world_csys.shape[1] 
    result = []
    for i in range(n):
        delta = np.linalg.norm(np.array([x, y, z]) - pts_world_csys[:,i])
        result.append(delta - pts_dist[i])
        
    return result

def compute_coord_from_triangule_locate(pts1_c1_csys, pt2_c1_csys,
                                        pts1_world_csys):
    """
    compute coordinate based on method 'triangule location' 
    
    """    
    
    pts_dist = compute_dist(pts1_c1_csys, pt2_c1_csys)
    sol = optimize.fsolve(compute_coord_from_triangule_locate_eqs, (1, 1, 1), 
                          factor=10, args=(pts1_world_csys, pts_dist))

    return sol, pts_dist

def compute_dist(pts1, target_pt):
    """
    compute the distance between pts1 and target_pt
    
    pts1: 3*N
    target_pt: 3*1
    """

    pts_dist = []
    n = pts1.shape[1]
    for i in range(n):
        pts_dist.append(np.linalg.norm(pts1[:,i] - target_pt))
    
    return np.array(pts_dist)


def compute_TM_from_four_points(pts1, pts2, is_shape3x4 = True):
    """
    computer the transfer matrix from pts1 to pts2
    based on two different coordinate systems
    T_M*pts1 = pts2
    """
    n = pts1.shape[1]
    if pts2.shape[1] != n:
        raise ValueError("Number of points don't match.")
    
    # build matrix for equations
    A = np.zeros((3*n,12))
    B = np.zeros((3*n,1))
    #AX=B
    for i in range(n):
        A[3*i] = [pts1[0,i], pts1[1,i], pts1[2,i], 1.0, \
                  0.0, 0.0, 0.0, 0.0, \
                  0.0, 0.0, 0.0, 0.0]
            
        A[3*i+1] = [0.0, 0.0, 0.0, 0.0,\
                 pts1[0,i], pts1[1,i], pts1[2,i], 1.0,
                 0.0, 0.0, 0.0, 0.0]

        A[3*i+2] = [0.0, 0.0, 0.0, 0.0,\
                 0.0, 0.0, 0.0, 0.0,\
                 pts1[0,i], pts1[1,i], pts1[2,i], 1.0]
    
        B[3*i] = pts2[0,i]
        B[3*i+1] = pts2[1,i]
        B[3*i+2] = pts2[2,i]
    # compute with least square solution
    A_m = inv(A.T.dot(A)).dot(A.T)
    #T vector
    T_M = A_m.dot(B).reshape((3,4))
    if not is_shape3x4:
        T_M = np.row_stack((T_M, np.array([0, 0, 0, 1])))
    #T_M = np.linalg.solve(A, B).reshape((3,4))
    return T_M
    

def comupute_R_t_from_fundamental(F, K0, K1):
    """
    essential matrix E = K1.T*F*K0 = RS
    using SVD to E, and get RS
    P0 = (I|0), P1 = (R|S)
    four solutions:
        (UWV.T | U(0,0,1).T)
        (UWV.T | -U(0,0,1).T)
        (UW.TV.T | U(0,0,1).T)
        (UW.TV.T | -U(0,0,1).T)
    Refer to 'Hartley: Estimation of Relative Camera Positions for 
        Uncalibrated Cameras'
    
    """

    E = K1.T.dot(F).dot(K0)
    solutions = comupute_R_t_from_essential(E)
    return solutions


def comupute_R_t_from_essential(E):
    """
    compute the possible Rotation matrixs and translation vectors
    based on essential matrix.
        
    """
    R_t_l = []

    U,S,VT = linalg.svd(E)

    
    W = np.array([[0, 1, 0], \
                   [-1, 0, 0], \
                   [0, 0, 1]])
    R = U.dot(W).dot(VT)
    if linalg.det(R) > 0:
        R1 = R
        R2 = R
        R3 = U.dot(W.T).dot(VT)
        R4 = R3
    else:
        R1 = -R
        R2 = -R
        R3 = -U.dot(W.T).dot(VT)
        R4 = R3  
    t1 = U.dot([0, 0, 1])
    t2 = -U.dot([0, 0, 1])
    t3 = U.dot([0, 0, 1])
    t4 = -U.dot([0, 0, 1])
    R_t_l.append((R1, t1))
    R_t_l.append((R2, t2))
    R_t_l.append((R3, t3))
    R_t_l.append((R4, t4))
    
    return R_t_l


def verify_correct_transf_m(proj_M1, proj_M2_list, img_pts1, img_pts2):
    """
    get the correct R_t
    choose the R_t which has maxinum positive z axis value
    """
    num_pz_ms = [] #number of positive z axis value with all possible 
    for proj_M2 in proj_M2_list:
        num_pz_ms.append(get_num_of_positive_z_axis(proj_M1, proj_M2, img_pts1, img_pts2))

    return num_pz_ms.index(max(num_pz_ms))

def get_num_of_positive_z_axis(proj_M1, proj_M2, img_pts1, img_pts2):    
    """
    1. the z axis value after restruct must be positive
    refer to paper 
    '基于自由运动一维标定物的多摄像机参数标定方法与实验'
    """    
    sum_positive_v = 0
    
    points_num = img_pts1.shape[1]
    for i in range(points_num): 
        coordinate_3d = triangulate_point(img_pts1[:,i],img_pts2[:,i],
                                          proj_M1, proj_M2)
        if coordinate_3d[2] >= 0:
            sum_positive_v +=1
    return sum_positive_v



def compute_scale_of_t(proj_M1, proj_M2, img_pts1, img_pts2, stand_len):
    """
    compute the scale of translation vector
    stand_len: the standard length of two points
        
    """
    scale_sum = 0
    counter = 0
    num_one_group = 3

    sample_num = img_pts1.shape[1]//num_one_group
         # the number of samples
    for i in range(sample_num):  
        pt_Blue = triangulate_point(img_pts1[:,3*i+1],img_pts2[:,3*i+1],proj_M1,proj_M2)
        pt_Green = triangulate_point(img_pts1[:,3*i+2],img_pts2[:,3*i+2],proj_M1,proj_M2)
        if pt_Blue[2] and pt_Green[2]:
            #z axis must be greater than 0
            counter += 1
            p2p_len = np.linalg.norm(pt_Blue-pt_Green)

            scale_sum += stand_len/p2p_len
    scale_sum /=  counter
    print("scale: %f, counter:%d \r\n"%(scale_sum, counter))
    return scale_sum

        


def compute_unique_R_t_from_essential(K1, K2, E, img_pts1, img_pts2, stand_len):
    """
    compute the correct Rotation matrix and translation vector.
    
    """
    proj_M1 = K1.dot(np.column_stack((np.eye(3, 3), np.zeros((3, 1)))))
    proj_M2_list = []
    R_t_l = comupute_R_t_from_essential(E)
    for R_t in R_t_l:
        proj_M2_list.append(K2.dot(np.column_stack(R_t)))
    idx = verify_correct_transf_m(proj_M1, proj_M2_list, img_pts1, img_pts2)
    R_t = R_t_l[idx]
    proj_M2 = proj_M2_list[idx]
    scale = compute_scale_of_t(proj_M1, proj_M2, img_pts1, img_pts2, stand_len)
    R_t = list(R_t)
    R_t[1] = scale*R_t[1]
    return tuple(R_t)    
            
            