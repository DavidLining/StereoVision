# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 10:17:45 2018

@author: Morgan.Li
"""
from calibration.calibrate import CameraCalibrator, StereoCalibrator, camera_database
import os,operator
import numpy as np
from utility.file_operation import store_result

#os.getcwd() 
def stereo_position_func(camera1_flag, camera2_flag, point_flag="Yellow", recalib = False, is_print = False, is_sync=True):
    coordinates_file_path = camera_database.coordinate_file
    last_coordinates_file_path = camera_database.last_coordinate_file
    sec_last_coordinates_file_path = camera_database.sencond_last_coordinate_file
    
    
    camera1_cal = CameraCalibrator(camera1_flag, os.getcwd()+r'/data', "pts_3d", "pts_2d", recalib)
    camera2_cal = CameraCalibrator(camera2_flag, os.getcwd()+r'/data', "pts_3d", "pts_2d", recalib)  
    
    stereo_cal = StereoCalibrator(camera1_cal.proj_mat, camera2_cal.proj_mat)
    camera1_data, camera2_data = stereo_cal.load( coordinates_file_path, camera1_flag, camera2_flag)

    #true value 
    true_coordinate_3d = camera1_data["3d"][point_flag]  
    
    point_2d_camera1 = camera1_data["2d"][point_flag].copy()
    point_2d_camera2 = camera2_data["2d"][point_flag].copy()
    error_3d, error = _stereo_cal(point_2d_camera1, point_2d_camera2, camera1_cal.proj_mat, camera2_cal.proj_mat, stereo_cal, true_coordinate_3d)
    if error_3d and error:    
        if(is_print):
            print("\nStart to restruct %s point with %s and %s..."%(point_flag, camera1_flag, camera2_flag))
            print("Truth 3d point coordinates:")
            print(true_coordinate_3d)    
            print("Restruct Error with same frame:")
            print(error_3d, error) 
    else:
        if(is_print):
            print("Point %s is not in %s or %s imaging area."%(point_flag, camera1_flag, camera2_flag))
        return error_3d, error
    if not is_sync:
        camera1_last_data, camera2_last_data = \
        stereo_cal.load( last_coordinates_file_path, camera1_flag, camera2_flag)
        
        camera1_sec_last_data, camera2_sec_last_data = \
        stereo_cal.load( sec_last_coordinates_file_path, camera1_flag, camera2_flag)
        point_2d_sec_last_camera2 = camera2_sec_last_data["2d"][point_flag].copy()
        
        point_2d_camera1 = camera1_data["2d"][point_flag].copy()
        point_2d_last_camera2 = camera2_last_data["2d"][point_flag].copy()
        delta = list(map(operator.sub, point_2d_last_camera2, point_2d_sec_last_camera2))
        point_2d_last_cal_camera2 =  list(map(operator.add, point_2d_last_camera2, delta))
        error_3d_no_cal, error_no_cal = _stereo_cal(point_2d_camera1, point_2d_last_camera2, camera1_cal.proj_mat, camera2_cal.proj_mat, stereo_cal, true_coordinate_3d)

        error_3d, error = _stereo_cal(point_2d_camera1, point_2d_last_cal_camera2, camera1_cal.proj_mat, camera2_cal.proj_mat, stereo_cal, true_coordinate_3d)
   
        if(is_print):   
                print("Restruct Error with no framing before cal:")
                print(error_3d_no_cal, error_no_cal)
                print("Restruct Error with no framing after cal:")
                print(error_3d, error)
                print("%s 2D point with different frame:"%camera2_flag)
                print(point_2d_camera2, point_2d_last_camera2,point_2d_last_cal_camera2)
    return error_3d, error
    
def _stereo_cal(point_2d_camera1, point_2d_camera2, proj_mat_camera1, proj_mat_camera2, stereo_cal, true_coordinate_3d):
    error_3d = None
    error = None
    if point_2d_camera1 and point_2d_camera2:
        point_2d_camera1 = np.array(point_2d_camera1)
        point_2d_camera2 = np.array(point_2d_camera2)
        #estimated value
        coordinate_3d = stereo_cal.calibrate(point_2d_camera1, point_2d_camera2, proj_mat_camera1, proj_mat_camera2)
        error_3d = list(map(lambda x,y:x - y,coordinate_3d,true_coordinate_3d))
        error = np.sqrt(sum([np.square(item) for item in error_3d]))
    return error_3d, error


def optimize_position_func(point_flag="Yellow", recalib = False, is_sync=True):
    camera_list = camera_database.camera_list
    camera_num = len(camera_list)
    camera_group_list = []
    error_dict = {}
    error_list = []
    for i in range(camera_num):
        if(camera_num - i >1):
            for j in range(i+1, camera_num):
                camera_group_list.append((camera_list[i], camera_list[j]))

    for camera_group in camera_group_list:
        error_3d, error = stereo_position_func(camera_group[0], camera_group[1], point_flag, recalib, is_print = False, is_sync = is_sync)
        if error_3d and error:
            error_dict[camera_group] = {"error_3d": error_3d, "error":error}
            error_list.append(error)
    error_min = min(error_list)
    camera_group_error_min = list(error_dict.keys())[error_list.index(error_min)]
    print("\nMinimal Error with Camera Group:")
    print(camera_group_error_min)
    print(error_dict[camera_group_error_min])
    return error_min



def get_error_from_feature_extract(tar_dis, pixel_size, focal_len, camera_dis):
    error = np.true_divide((np.square(tar_dis)*pixel_size)/(focal_len*camera_dis + tar_dis*pixel_size))
    return error

#optimize_position_func()