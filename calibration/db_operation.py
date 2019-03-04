# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 15:43:10 2018

@author: Morgan.Li
"""
import numpy as np
from numpy.linalg import inv


def get_ext_params_from_db(camera1_flag, camera2_flag, params_db):
    key = camera1_flag+"_"+camera2_flag   
    R_T = params_db.get_data(key+"_ext_camera_csys") 
    if not R_T:
        inv_key = camera2_flag+"_"+camera1_flag     
        R_T_inv= params_db.get_data(inv_key+"_ext_camera_csys") 
        if R_T_inv:
            R_T_inv = np.array(R_T_inv).reshape((3,4))
            R_T_inv = np.row_stack((R_T_inv, np.array([0,0,0,1])))
            return inv(R_T_inv)[0:3,:]
        else:
            return None
    else:
        return np.array(R_T).reshape((3,4))