# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 14:25:54 2018

@author: Morgan.Li
"""

from pylab import plot
import numpy as np
from scipy import linalg
from numpy.linalg import inv, solve

def compute_P(x,X):
    """    Compute camera matrix from pairs of
        2D-3D correspondences (in homog. coordinates). """

    n = x.shape[1] #column of matrix
    if X.shape[1] != n:
        raise ValueError("Number of points don't match.")
        
    # create matrix for DLT solution
    M = np.zeros((3*n,12+n))
    for i in range(n):
        M[3*i,0:4] = X[:,i]
        M[3*i+1,4:8] = X[:,i]
        M[3*i+2,8:12] = X[:,i]
        M[3*i:3*i+3,i+12] = -x[:,i]
        
    U,S,V = linalg.svd(M)
    
    return V[-1,:12].reshape((3,4))


def triangulate_point_SVD(x1,x2,P1,P2):
    """ Point pair triangulation from 
        SVD solution.
        Slower than 'triangulate_point'
        """
        
    M = np.zeros((6,6))
    M[:3,:4] = P1
    M[3:,:4] = P2
    M[:3,4] = -x1
    M[3:,5] = -x2

    U,S,V = linalg.svd(M)
    X = V[-1,:4]

    return (X / X[3])[0:3]

def triangulate_point_opt(x1, x2, P1, P2):
    u1, v1 = x1[0:2]
    u2, v2 = x2[0:2]
    p1_31, p1_32, p1_33 = P1[2, 0:3]
    p2_31, p2_32, p2_33 = P2[2, 0:3]
    A1 = np.array([[u1*p1_31, u1*p1_32, u1*p1_33], \
                  [v1*p1_31, v1*p1_32, v1*p1_33], \
                  [u2*p2_31, u2*p2_32, u2*p2_33], \
                  [v2*p2_31, v2*p2_32, v2*p2_33]])
    A2 = np.row_stack((P1[:2,:3], P2[:2,:3]))
    A = A1 - A2
    B1 = P1[0:2,3] -  P1[2,3]*x1[0:2]
    B2 = P2[0:2,3] -  P2[2,3]*x2[0:2]
    B = np.concatenate((B1, B2)).reshape((4,1))

    A_T = A.T
    pts_ref_csys = inv(A_T.dot(A)).dot(A_T).dot(B)
    
    return pts_ref_csys.T[0]

def diag_block_mat_slicing(L, N, shp):
    L = L.reshape(N, shp[0], shp[1])
    r = np.arange(N)
    out = np.zeros((N,shp[0],N,shp[1]))
    out[r,:,r,:] = L
    return out.reshape(np.asarray(shp)*N)


def triangulate_pts_opt(X1, X2, P1, P2, stand_lens, sub_size):
     
    X1_Origin = X1[:2]
    X2_Origin = X2[:2]
    pts_num = X1.shape[-1]
    error1_array =  np.zeros((1, pts_num))[0]
    error2_array =  np.zeros((1, pts_num//3))[0]  
    divider, remainder = np.divmod(pts_num,sub_size)
    for i in np.arange(divider+1):
        start_col = sub_size*i 
        
        if i == divider:
            step_size = remainder
        else:
            step_size = sub_size
        X1 = X1_Origin[:, start_col:start_col+step_size]
        X2 = X2_Origin[:, start_col:start_col+step_size]
        X = np.row_stack((X1, X2))
        pts_num = X1.shape[-1]
        X_one_col = X.T.reshape((4*pts_num,1))    
    
        p1_31, p1_32, p1_33 = P1[2, 0:3]
        p2_31, p2_32, p2_33 = P2[2, 0:3]
        A2 = np.array([[p1_31, p1_32, p1_33], \
                       [p1_31, p1_32, p1_33], \
                       [p2_31, p2_32, p2_33], \
                       [p2_31, p2_32, p2_33]])
        A1 = np.repeat(X_one_col,3, axis=1)
        A2 = np.tile(A2, (pts_num, 1))
        
        A3 = np.row_stack((P1[:2,:3], P2[:2,:3]))
        A3 = np.tile(A3, (pts_num, 1))
    
        A = A1*A2 - A3
        A = diag_block_mat_slicing(A, pts_num, (4,3))   
        B1 = np.row_stack((P1[0:2,3], P2[0:2,3])).reshape((4,1))
        B1 = np.tile(B1, (pts_num, 1))    
        B2 = np.row_stack((P1[2,3]* X1, P2[2,3]* X2))  
        B2 = B2.T.reshape((4*pts_num,1))      
        B = B1 - B2  
        A_T = A.T
 
        pts_ref_csys = (inv(A_T.dot(A)).dot(A_T).dot(B)).reshape((pts_num//3, 9))
        pts_move_ref_csys = np.column_stack((pts_ref_csys[:, 6:9], \
                                             pts_ref_csys[:, 0:6]))
        delta_p2p = pts_ref_csys-pts_move_ref_csys
        dist = np.linalg.norm(delta_p2p.reshape(pts_num, 3), axis=1)
        dist_error = dist - np.tile(stand_lens, pts_num//3)
        error1_array[start_col:start_col+step_size] =  dist_error
        dist_3cols = dist.reshape((pts_num//3,3))
        error2_array[start_col//3:(start_col+step_size)//3] = \
                    dist_3cols[:,2] - dist_3cols[:,0] - dist_3cols[:,1]

    return error1_array, error2_array


def triangulate_point(x1, x2, P1, P2):
    """ Point pair triangulation from 
        least squares solution. """

    A1 = x1[0:2].reshape((2,1)).dot(P1[2,:3].reshape((1,3))) \
        - P1[:2,:3]
    A2 = x2[0:2].reshape((2,1)).dot(P2[2,:3].reshape((1,3))) \
        - P2[:2,:3]
    A = np.row_stack((A1,A2))
    B1 = P1[0:2,3] -  P1[2,3]*x1[0:2]
    B2 = P2[0:2,3] -  P2[2,3]*x2[0:2]

    B = np.concatenate((B1, B2)).reshape((4,1))
    #M vector, not include m34
    A_T = A.T
    pts_ref_coord = inv(A_T.dot(A)).dot(A_T).dot(B)
    return pts_ref_coord.T[0]
    
def triangulate_multi_cameras(X, P):
    """    
    Mutiple-view triangulation of points in 
    x1,x2,x3.... with projection matrix p1, p2, p3...
    """
    if len(X) == len(P):
        num = len(X)
        A = np.zeros((2*num, 3))
        B = np.zeros((2*num, 1))
        for idx in np.arange(len(X)):
            x = X[idx]
            p = P[idx]
            
            A[2*idx:2*idx+2,] = x[0:2].reshape((2,1)).dot(p[2,:3].reshape((1,3))) \
                - p[:2,:3]  
            
            B[2*idx:2*idx+2,] = (p[0:2,3] -  p[2,3]*x[0:2]).reshape((2,1))
         
            
        pts_ref_coord = inv(A.T.dot(A)).dot(A.T).dot(B)
        return pts_ref_coord.T[0]
    else:
        raise ValueError("Number of points don't match.")

def compute_fundamental(x1,x2):
    """    Computes the fundamental matrix from corresponding points 
        (x1,x2 3*n arrays) using the 8 point algorithm.
        Each row in the A matrix below is constructed as
        [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1] 
        assume: x2.T.dot(F).x1 = 0
        """
    
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")
    
    # build matrix for equations
    A = np.zeros((n,9))
    for i in range(n):
#        A[i] = [x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i]*x2[2,i],
#                x1[1,i]*x2[0,i], x1[1,i]*x2[1,i], x1[1,i]*x2[2,i],
#                x1[2,i]*x2[0,i], x1[2,i]*x2[1,i], x1[2,i]*x2[2,i] ]
        A[i] = [x2[0,i]*x1[0,i], x2[0,i]*x1[1,i], x2[0,i]*x1[2,i],
                x2[1,i]*x1[0,i], x2[1,i]*x1[1,i], x2[1,i]*x1[2,i],
                x2[2,i]*x1[0,i], x2[2,i]*x1[1,i], x2[2,i]*x1[2,i] ]            
    # compute linear least square solution
    U,S,V = linalg.svd(A)
    F = V[-1].reshape(3,3)
        
    # constrain F
    # make rank 2 by zeroing out last singular value
    U,S,V = linalg.svd(F)
    S[2] = 0
    F = np.dot(U,np.dot(np.diag(S),V))
    return F/F[2,2]


def compute_epipole(F):
    """ Computes the (right) epipole from a 
        fundamental matrix F. 
        (Use with F.T for left epipole.) """
    
    # return null space of F (Fx=0)
    U,S,V = linalg.svd(F)
    e = V[-1]
    return e/e[2]
 



def compute_focal_len(F):
    """
    compute focal length based on Hartley algorithm
    
    """
    e0 = compute_epipole(F)
    e1 = compute_epipole(F.T)
    #print("Fundamental: \r\n", F)
    denom_e0 = np.sqrt(e0[0]**2 + e0[1]**2)
    denom_e1 = np.sqrt(e1[0]**2 + e1[1]**2)    
    #e1.T*F*e0 = 0
    T0 = np.array([[e0[0]/denom_e0, e0[1]/denom_e0, 0], \
                    [e0[1]/denom_e0, -e0[0]/denom_e0, 0], \
                    [0, 0, 1]])
    T1 = np.array([[e1[0]/denom_e1, e1[1]/denom_e1, 0], \
                    [e1[1]/denom_e1, -e1[0]/denom_e1, 0], \
                    [0, 0, 1]])    
    t_e0 = T0.dot(e0.T)  #e0 after translate
    t_e1 = T1.dot(e1.T)  #e1 after translate
    Fe = inv(T1.T).dot(F).dot(inv(T0))

    Fe_left = np.diag([t_e1[2], 1, -t_e1[0]])
    Fe_right = np.diag([t_e0[2], 1, -t_e0[0]])
    Fe_center = inv(Fe_left).dot(Fe).dot(inv(Fe_right))

    c1 = Fe_center[0,0]
    c2 = Fe_center[0,1]
    c3 = Fe_center[1,0]
    c4 = Fe_center[1,1]

    f0 = (-c1*c3*t_e0[0]**2)/(c1*c3*t_e0[2]**2 + c2*c4)
    f1 = (-c1*c2*t_e1[0]**2)/(c1*c2*t_e1[2]**2 + c3*c4)
    f0 = np.sqrt(f0)
    f1 = np.sqrt(f1)
    return f0, f1
    
    
def plot_epipolar_line(im,F,x,epipole=None,show_epipole=True):
    """ Plot the epipole and epipolar line F*x=0
        in an image. F is the fundamental matrix 
        and x a point in the other image."""
    
    m,n = im.shape[:2]
    line = np.dot(F,x)
    
    # epipolar line parameter and values
    t = np.linspace(0,n,100)
    lt = np.array([(line[2]+line[0]*tt)/(-line[1]) for tt in t])

    # take only line points inside the image
    ndx = (lt>=0) & (lt<m) 
    plot(t[ndx],lt[ndx],linewidth=2)
    
    if show_epipole:
        if epipole is None:
            epipole = compute_epipole(F)
        plot(epipole[0]/epipole[2],epipole[1]/epipole[2],'r*')
    

def skew(a):
    """ Skew matrix A such that a x v = Av for any v. """

    return np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])


def compute_P_from_fundamental(F):
    """    Computes the second camera matrix (assuming P1 = [I 0]) 
        from a fundamental matrix. """
        
    e = compute_epipole(F.T) # left epipole
    Te = skew(e)
    return np.vstack((np.dot(Te,F.T).T,e)).T


def compute_P_from_essential(E):
    """    Computes the second camera matrix (assuming P1 = [I 0]) 
        from an essential matrix. Output is a list of four 
        possible camera matrices. """
    
    # make sure E is rank 2
    U,S,V = linalg.svd(E)
    if np.det(np.dot(U,V))<0:
        V = -V
    E = np.dot(U,np.dot(np.diag([1,1,0]),V))    
    
    # create matrices (Hartley p 258)
    Z = skew([0,0,-1])
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    
    # return all four solutions
    P2 = [np.vstack((np.dot(U,np.dot(W,V)).T,U[:,2])).T,
             np.vstack((np.dot(U,np.dot(W,V)).T,-U[:,2])).T,
            np.vstack((np.dot(U,np.dot(W.T,V)).T,U[:,2])).T,
            np.vstack((np.dot(U,np.dot(W.T,V)).T,-U[:,2])).T]

    return P2


def compute_fundamental_normalized(x1,x2):
    """    Computes the fundamental matrix from corresponding points 
        (x1,x2 3*n arrays) using the normalized 8 point algorithm.
        Assume: X2.T.dot(F).dot(X1) = 0
    """

    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")

    # normalize image coordinates
    x1 = x1 / x1[2]
    mean_1 = np.mean(x1[:2],axis=1)
    
    variance_1 = np.sum(np.square(x1[0,:] - mean_1[0])) + \
                    np.sum(np.square(x1[1,:] - mean_1[1])) 
    variance_1 /= n                
    S1 = np.sqrt(2) / variance_1
    T1 = np.array([[S1,0,-S1*mean_1[0]],[0,S1,-S1*mean_1[1]],[0,0,1]])
    x1 = np.dot(T1,x1)
    
    x2 = x2 / x2[2]
    mean_2 = np.mean(x2[:2],axis=1)
    variance_2 = np.sum(np.square(x2[0,:] - mean_2[0])) + \
                    np.sum(np.square(x2[1,:] - mean_2[1]))     
    variance_2 /= n 
    S2 = np.sqrt(2) / variance_2
    T2 = np.array([[S2,0,-S2*mean_2[0]],[0,S2,-S2*mean_2[1]],[0,0,1]])
    x2 = np.dot(T2,x2)
    # compute F with the normalized coordinates
    F = compute_fundamental(x1,x2)

    # reverse normalization
    F = np.dot(T2.T,np.dot(F,T1))

    return F/F[2,2]


def compute_E_from_fundamental(F, K1, K2):
    """
    Using linear algorithm method to compute the Essential matrix from 
    fundamental matrix and internal parameter matrix K1 and K2.
    (assuming the external matrix of camera1 = [I 0]) 
    Since F has seven freedom degrees and E has five freedom degrees,
    using linear algorithm to compute E is not a good method
       Assume: X2.T.dot(F).dot(X1) = 0
    """
    E = K2.T.dot(F).dot(K1)
    return E

