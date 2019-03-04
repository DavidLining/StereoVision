# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 13:44:23 2018

@author: Morgan.Li
"""
import numpy as np
from cv2 import Rodrigues
from calib_config import len_relation, scale_hw
from scipy.optimize import minimize, least_squares
import time
from db_lib.database import camerasParamsDatabase
from scipy.sparse import lil_matrix

def multi_reprojerr(x,mask,img_filter_cameras, pts_num, cameras_num, \
                    observations_num, camera_ref_pts_indices):
 
    """
    reprojection error used for bundle adjustment
    """
    proj_cameras = np.zeros((cameras_num, 3, 4))
    
    error_array = np.zeros((1, observations_num*2))[0]    
    
    proj_cameras = np.array(x[:12*cameras_num]).reshape((cameras_num, 3, 4))
    
    ref_re_cameras = np.array(x[12*cameras_num:]).reshape((3, pts_num))
    ref_re_cameras = np.row_stack((ref_re_cameras, np.ones((1,pts_num))))
    start_row = 0
    for c_idx in np.arange(cameras_num):
        pts_indices = camera_ref_pts_indices[c_idx]
        rows_size = 2*np.size(pts_indices)
        img_pts_re = proj_cameras[c_idx, :, :].dot(ref_re_cameras[:,pts_indices])
        img_pts_re = img_pts_re/img_pts_re[-1,:]
        error_m = img_filter_cameras[c_idx,:2,pts_indices].T - img_pts_re[:2,:]
        error_array[start_row:start_row+rows_size] = error_m.T.reshape((1,rows_size))
        start_row +=rows_size
    return error_array

def bundle_adjustment_sparsity(mask, pts_num, cameras_num, observations_num, \
                               camera_ref_pts_indices):
    """
    sparse matrix used for bundle adjustment
    """
    n = cameras_num * 12 + pts_num * 3
    A = lil_matrix((observations_num*2, n), dtype=int)
    start_row = 0
    for c_idx in np.arange(cameras_num):
        pts_indices = camera_ref_pts_indices[c_idx]
        rows_size = 2*np.size(pts_indices)
        col_re_pts = cameras_num*12+3*pts_indices
#        cols = np.concatenate((col_re_pts,\
#                               col_re_pts+1,\
#                               col_re_pts+2))
        rows = np.arange(start_row, start_row+rows_size)
        A[rows, c_idx*12:c_idx*12+12] = 1
           
        cols = np.repeat(col_re_pts,2)        
        indexes = (rows,cols)
        A[indexes] = 1

        cols = np.repeat(col_re_pts+1, 2)
        indexes = (rows,cols)
        A[indexes] = 1

        cols = np.repeat(col_re_pts+2, 2)
        indexes = (rows,cols)
        A[indexes] = 1
        
        start_row +=rows_size
        
    
    return A



def bundle_adjust(init_params,bridge_camera, mask,img_filter_cameras, \
                  pts_num, cameras_num, camera_list, is_restart=False):
    """
     bundle adjustment
    """

    params_BA = camerasParamsDatabase(bridge_camera +"_Params_BA.db")
    if is_restart or params_BA.is_db_null():
        t0 = time.time()
        camera_ref_pts_indices = []
        observations_num = 0
        for c_idx in np.arange(cameras_num):
            pts_idxes = np.where(mask[c_idx, :] > 0)[0] 
            observations_num += np.size(pts_idxes)
            camera_ref_pts_indices.append(pts_idxes)
        

        A = bundle_adjustment_sparsity(mask, pts_num, cameras_num, observations_num, \
                                       camera_ref_pts_indices)        
        
        
        
        params = least_squares(multi_reprojerr, init_params, jac_sparsity=A, \
                               verbose=0, x_scale='jac', ftol=1e-8, method='trf',
                               args=(mask,img_filter_cameras, pts_num, cameras_num, \
                                     observations_num, camera_ref_pts_indices)) 
        t1 = time.time()
        print("Bundle Adjustment take {0:.0f} seconds".format(t1 - t0))
        proj_cameras_list = params.x[:96]
        print("Opt status:", params.message, params.success)   
        for c_idx in np.arange(cameras_num):
            params_BA.store_data(camera_list[c_idx], proj_cameras_list[12*c_idx:12*(c_idx+1)].tolist()) 
            
        return params.x

    
    
    
    
    
    

