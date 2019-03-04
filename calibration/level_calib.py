# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 09:24:10 2018

@author: Morgan.Li
"""


from calibration.calibrate import CameraCalibDataCollector, camera_database
from db_lib.database import LevelCalibDB, camerasParamsDatabase
from calibration.stereo_calib import coordinate_sys_move
import numpy as np 
from calibration.external_params import compute_coord_from_triangule_locate,\
                                        compute_TM_from_four_points
from calibration.internal_params import triangulate_multi_cameras
from calibration.calib_config import camera_list
from calibration.db_operation import get_ext_params_from_db

def level_calib_data_collect(camera_list):
    """collect data for level calibration"""
    
    data_collector = CameraCalibDataCollector()
    data_collector.data_collect(camera_database.coordinate_file,\
                                camera_list, is_level=True)           
    print("Collect data for level calibration")
    
    
def get_level_calib_data(camera_list, img_accuracy=None):
    """
    get the initial data used for level calibration from database 
    """
    cameras_img_coord = []
    level_data_db = LevelCalibDB()
    for camera in camera_list:
        camera_2d = level_data_db.get_data(camera)["2d"]
        for key in camera_2d.keys():
            coordinate_sys_move(camera_2d[key], accuracy = img_accuracy)
        cameras_img_coord.append(camera_2d)
    cameras_3d_coord = level_data_db.get_data(camera_list[0])["3d"]
    return cameras_img_coord, cameras_3d_coord


def get_scale_cameras(bridge_camera, camera_list, cameras_graph):
    """
    get the scales of other camera's coordinates during compute the 
    transfer matrix
    
    """
    cameras_graph.get_border_cost(bridge_camera)
    if bridge_camera in camera_list:
        camera1_idx = camera_list.index(bridge_camera)
        other_cameras = camera_list[:camera1_idx] + camera_list[camera1_idx+1:]
    else:
        other_cameras = camera_list
    cost_list = []
    cost_sum = 0
    for camera2_flag in other_cameras:
        cost = 1/cameras_graph.dijkstra(camera2_flag, bridge_camera)[0]    
        cost_list.append(cost)
        cost_sum += cost
        
    scales = [x/cost_sum for x in cost_list]
    return scales

def level_calibrate(bridge_camera, data_processor, img_accuracy=None):
    """
    level calibration
    
    """ 
    scales = get_scale_cameras(bridge_camera, camera_list, data_processor.cameras_graph)
    params_db = camerasParamsDatabase()   
    f0 = params_db.get_data(bridge_camera+"_int")

    K1 = np.array([[f0, 0, 0], [0, f0, 0], [0, 0, 1]])
    proj_M1 = K1.dot(np.column_stack((np.eye(3, 3), np.zeros((3, 1)))))

    
    res_pts_3d, ref_pts_3d = get_other_cameras_coord(bridge_camera,\
                                          camera_list = camera_list, img_accuracy=img_accuracy)

#    res_pts_3d, ref_pts_3d = get_other_cameras_coord(bridge_camera, camera2_flag)
    transfer_m = np.zeros((4,4))
    for i in np.arange(res_pts_3d.shape[1])[3:]:
        ref_pts_3d_ = np.column_stack((ref_pts_3d[:,0:3], ref_pts_3d[:,i])) 
        res_pts_3d_ = np.column_stack((res_pts_3d[:,0:3], res_pts_3d[:,i])) 
        transfer_m += scales[i-3]*compute_TM_from_four_points(ref_pts_3d_, res_pts_3d_, False)
    
    proj_M1_world_csys = proj_M1.dot(transfer_m)
    params_db.store_data(bridge_camera+"_params_world_csys", proj_M1_world_csys.tolist())
    update_proj_matrix(bridge_camera, transfer_m, camera_list)    

    
def update_proj_matrix(bridge_camera, transfer_m, camera_list):
    params_db = camerasParamsDatabase()  
    camera1_idx = camera_list.index(bridge_camera)
    other_cameras = camera_list[:camera1_idx] + camera_list[camera1_idx+1:]
    for camera2_flag in other_cameras:    
        rt_key = bridge_camera+"_"+camera2_flag     
        ext_camera_csys = get_ext_params_from_db(bridge_camera, camera2_flag, params_db)
        f = params_db.get_data(camera2_flag+"_int")        
        if ext_camera_csys is not None and f:
            K = np.array([[f, 0, 0], [0, f, 0], [0, 0, 1]])
            proj_M2 = K.dot(ext_camera_csys).dot(transfer_m)
            params_db.store_data(rt_key +"_params_world_csys", proj_M2.tolist())   



def get_other_cameras_coord(bridge_camera, camera2_flag=None, \
                            camera_list = None, img_accuracy=None):
    """
    get cameras origin coordinates in bridge_camera coordinate system
    """
    params_db = camerasParamsDatabase()  
    if camera2_flag:
        other_cameras = [camera2_flag]
    elif camera_list:
        if bridge_camera in camera_list:
            camera1_idx = camera_list.index(bridge_camera)
            other_cameras = camera_list[:camera1_idx] + camera_list[camera1_idx+1:]
        else:
            other_cameras = camera_list
    proj_list = []    
    flags = ("Green", "Blue", "Yellow")
    img_pts_data = {"Green":[], "Blue":[], "Yellow":[]}
    res_pts_3d= np.array([])
    ref_pts_3d = []    
    camera_pts_list = []
    for camera2_flag in other_cameras:
        camera2_idx = other_cameras.index(camera2_flag)              
        if camera2_idx ==0:
            f0 = params_db.get_data(bridge_camera+"_int")
            K1 = np.array([[f0, 0, 0], [0, f0, 0], [0, 0, 1]])
            proj_M1 = K1.dot(np.column_stack((np.eye(3, 3), np.zeros((3, 1)))))
            proj_list.append(proj_M1)
        f1 = params_db.get_data(camera2_flag+"_int")
        K2 = np.array([[f1, 0, 0], [0, f1, 0], [0, 0, 1]])
        R_T = get_ext_params_from_db(bridge_camera, camera2_flag, params_db) 
        R_M = R_T[:,0:3]
        t = R_T[:,3]
        c2_origin_c1_csys = np.linalg.solve(R_M, -t)        
        if not np.allclose(np.dot(R_M, c2_origin_c1_csys), -t):
            raise ValueError("Can not get the Camera coordinate in CSYS.")
        camera_pts_list.append(c2_origin_c1_csys)
        proj_M2 = K2.dot(np.column_stack((R_M, t)))
        proj_list.append(proj_M2)

        img_coord_pts, ref_coord_pts = get_level_calib_data([bridge_camera, camera2_flag], \
                                                            img_accuracy)
        camera1_ref_2d, camera2_ref_2d = img_coord_pts
        for flag in flags:
            if camera2_idx ==0:
                ref_pts_3d.append(ref_coord_pts[flag])
                img_pts_data[flag].append(np.array(camera1_ref_2d[flag]))                
            img_pts_data[flag].append(np.array(camera2_ref_2d[flag]))


    for flag in flags:
        res_pt_3d = triangulate_multi_cameras(img_pts_data[flag], proj_list)
        res_pts_3d = np.append(res_pts_3d, res_pt_3d)        
    new_res_pts_3d = res_pts_3d = res_pts_3d.reshape((3, 3)).T
    new_ref_pts_3d = ref_pts_3d = np.array(ref_pts_3d).T
    for camera2_flag in other_cameras:
        c2_origin_c1_csys = camera_pts_list[other_cameras.index(camera2_flag)]
        sol, pts_dist = compute_coord_from_triangule_locate(res_pts_3d, \
                                                            c2_origin_c1_csys, \
                                                            ref_pts_3d)   
#        print("Sol:", sol)        
#        sol = [round(x) for x in sol]
#        print("Round Sol:", sol)
#        error = compute_coord_from_triangule_locate_eqs(sol, ref_pts_3d, pts_dist)
#        print(error)
#        sol = np.array([-4.0, 8.0, -4.0])
        new_res_pts_3d = np.column_stack((new_res_pts_3d, c2_origin_c1_csys))
        new_ref_pts_3d = np.column_stack((new_ref_pts_3d, sol))    
    return new_res_pts_3d, new_ref_pts_3d        


