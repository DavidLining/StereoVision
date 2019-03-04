# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 11:45:28 2018

@author: Morgan.Li
"""
from db_lib import database
from calibration.ransac_five import ransac_five
import numpy as np
from calibration.external_params import compute_unique_R_t_from_essential
from calibration.internal_params import compute_fundamental_normalized, compute_focal_len, \
                                        compute_E_from_fundamental
from calibration.calib_config import u0, v0, len_relation, \
                            default_focal, focal_scale
from calibration.optimize_params import optimize_params_two_camera
from cv2 import Rodrigues



    

class stereo_data_processor(object):

    def __init__(self, camera1_flag, camera2_flag):
        self.camera = (camera1_flag, camera2_flag)
        self.points_flag = ("Red", "Blue", "Green")
 

    def load(self):
        img_data_camera1 = []
        img_data_camera2 = []
        img_init_data_camera1 = []
        img_init_data_camera2 = []  
        ref_init_data = []
        ref_filter_data = []
        camera1_db = database.CameraCalDatabase(self.camera[0])
        camera2_db = database.CameraCalDatabase(self.camera[1])  
        if len(camera1_db.db.getall()) == len(camera2_db.db.getall()):
            keys = list(camera1_db.db.getall())
            for key in keys[:]:
                img_coord_camera1 = camera1_db.get_cal_data(key)['2d']
                img_coord_camera2 = camera2_db.get_cal_data(key)['2d']
                ref_coord_camera1 = camera1_db.get_cal_data(key)['3d']               
                ref_coord_camera2 = camera2_db.get_cal_data(key)['3d']
                #make sure the data is not empty dict
                if (ref_coord_camera1 == ref_coord_camera2) and \
                    (len(set(img_coord_camera1.keys()))==4) and \
                    (len(set(img_coord_camera2.keys()))==4):
                    # if the image contains required three points
                    if set(self.points_flag).issubset(set(img_coord_camera1.keys())): 
                        for point_flag in img_coord_camera1.keys():
                            c1 = coordinate_sys_move(img_coord_camera1[point_flag], u0, v0)
                            c2 = coordinate_sys_move(img_coord_camera2[point_flag], u0, v0)
                            if point_flag in self.points_flag:
                                #red blue green
                                img_data_camera1.append(c1)
                                img_data_camera2.append(c2)
                                ref_filter_data.append(ref_coord_camera1[point_flag])
                        img_init_data_camera1.append(img_coord_camera1)
                        img_init_data_camera2.append(img_coord_camera2)
                        ref_init_data.append(ref_coord_camera1)
            return (np.array(img_data_camera1).T, np.array(img_data_camera2).T), \
                (img_init_data_camera1, img_init_data_camera2),\
                (ref_init_data, ref_filter_data)
        else:
            raise ValueError("Invalid Camera data(different size)")

 
def get_params_from_stereo_vision(img_data_camera1, img_data_camera2,\
                                  focals=None, data_size = 200):
    
    """
    get internal and external parameters from stereo vision 
    """
    op_x = None
    if img_data_camera1.shape[-1] > data_size:
        img_c1_computeF = img_data_camera1[:,:data_size]
        img_c2_computeF = img_data_camera2[:,:data_size]
    else:
        img_c1_computeF = img_data_camera1
        img_c2_computeF = img_data_camera2
    F = compute_fundamental_normalized(img_c1_computeF, img_c2_computeF)
    if not focals:
        #print(img_data_camera1, img_data_camera2)
        f0, f1 = compute_focal_len(F)
    else:
        f0, f1 = focals
    focal_min = default_focal -5
    focal_max = default_focal +5
    if not(f0>focal_min and f0<focal_max and f1>focal_min and f1<focal_max):
        f0 = f1 = default_focal
    print("Focals: ", f0, f1)        
    K1 = np.array([[f0, 0, 0], [0, f0, 0], [0, 0, 1]])
    K2 = np.array([[f1, 0, 0], [0, f1, 0], [0, 0, 1]])
#    X1 = img_data_camera1[:, 0:600]
#    X2 = img_data_camera2[:, 0:600]
#    try:
#        bestE = ransac_five(X1, X2, K1, K2, 1000, 0.0005)
##       print("Points Index: \r\n", bestE[0])
#        E_matrix = bestE[1]
##        print("Best essential from five point algorithm: \r\n", E_matrix)
#    except InvalidDenomError:
#        E_matrix = compute_E_from_fundamental(F, K1, K2)        
##        print("Essential from fundenmental matrix: \r\n", E_matrix)    

    E_matrix = compute_E_from_fundamental(F, K1, K2)     
    R_t = compute_unique_R_t_from_essential(K1, K2, E_matrix,\
                                            img_data_camera1, img_data_camera2, 1.0)
    
    R_v = Rodrigues(R_t[0])[0].T
    t_v = R_t[1]  
    init_x = []
    if not focals:
        init_x.extend([f0, f1])
    init_x.extend(R_v.tolist()[0])
    init_x.extend(t_v.tolist())

#    print("init params: ", init_x)
    
    op_x = optimize_params_two_camera(init_x, img_data_camera1, \
                                      img_data_camera2, len_relation, focals)
 #   print("optimize params: ", op_x) 


    return op_x
        


def coordinate_sys_move(img_3d, u0=u0, v0=v0, accuracy=None, focal_scale=focal_scale):
    """
    accuracy: ndigits precision after the decimal point
    """
    
    img_3d[0] = focal_scale*img_3d[0] - u0
    img_3d[1] = focal_scale*img_3d[1] - v0   

    if accuracy:
        img_3d[0] = round(img_3d[0],accuracy)
        img_3d[1] = round(img_3d[1],accuracy)  
        
 

    return  img_3d   




    