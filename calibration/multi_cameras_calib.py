# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 14:58:38 2018

@author: Morgan.Li
"""

from calibration.calib_config import len_relation, camera_list
import itertools
from calibration.dijkstra import Graph
import numpy as np
from calibration.stereo_calib import get_params_from_stereo_vision, \
                                    coordinate_sys_move
from db_lib.database import camerasParamsDatabase, CameraCalDatabase
from calibration.internal_params import triangulate_multi_cameras
from cv2 import Rodrigues
from calibration.bundle_adjustment import bundle_adjust
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
from calibration.level_calib import level_calibrate
from numpy.linalg import inv
from calibration.db_operation import get_ext_params_from_db
from utility.file_operation import generate_report



class multi_cameras_data_processor(object):

    def __init__(self, camera_list, sample_num =None, start_idx=0 , img_accuracy=None):
        if not sample_num:
            camera_db = CameraCalDatabase("camera1")
            sample_num =3*len(camera_db.db.getall())
            
        self.img_accuracy = img_accuracy
        self.camera_list = camera_list
        self.cameras_num = len(camera_list)
        self.points_flag = ("Red", "Blue", "Green")
        self.sample_num = sample_num
        self.mask = np.zeros((self.cameras_num, self.sample_num))  
        self.img_filter_cameras = np.zeros((self.cameras_num, 3, sample_num))
        self.ref_filter_cameras = np.zeros((self.cameras_num, 3, sample_num))
        #ref coordinates after reconstruct
        self.error_ref_pts = None
        self.ref_re_cameras = None
        self.ref_re_cameras_list = None
        
        self.proj_cameras_M = np.zeros((self.cameras_num, 3, 4))
        self.proj_cameras_list = None
        self.cameras_graph = Graph()

        self.load(start_idx)
        self.cal_cameras_graph()
      
    def load(self, start_idx):
        for camera in self.camera_list:
            c_idx = camera_list.index(camera) #camera index
            camera_db = CameraCalDatabase(camera)
            keys = list(camera_db.db.getall())
            if len(keys) >= self.sample_num/3:
                p_idx = 0 #point idx
                for i in np.arange(int(self.sample_num/3)):
                    key = keys[i]
                    if int(key) > start_idx:
                        img_coord_camera = camera_db.get_cal_data(key)['2d']
                        ref_coord_camera = camera_db.get_cal_data(key)['3d']
                        #make sure the data is not empty dict
                        # if the image contains required three points
                        if (len(set(img_coord_camera.keys()))==4) and \
                            set(self.points_flag).issubset(set(img_coord_camera.keys())):
                                
                            for point_flag in img_coord_camera.keys():
                                c1 = coordinate_sys_move(img_coord_camera[point_flag], \
                                                         accuracy = self.img_accuracy)
                                if point_flag in self.points_flag:
                                    #red blue green
                                    self.mask[c_idx, p_idx] = 1 
                                    self.img_filter_cameras[c_idx,:,p_idx] = c1
                                    self.ref_filter_cameras[c_idx,:,p_idx] = ref_coord_camera[point_flag]                            
                                    p_idx +=1
                        else:
                            p_idx +=3
        for p_idx in np.arange(self.sample_num):
            
            if(np.count_nonzero(self.mask[:, p_idx])<2):    
                self.mask[:, p_idx] = np.zeros((1, self.cameras_num))
                
        cols_allzero = np.where(~self.mask.any(axis=0))[0]
        #remove outliers, the pts not in any images
        self.mask = np.delete(self.mask, cols_allzero, axis=1)
        self.img_filter_cameras = np.delete(self.img_filter_cameras, \
                                            cols_allzero, axis=2)
        self.ref_filter_cameras = np.delete(self.ref_filter_cameras, \
                                            cols_allzero, axis=2)
        
        self.sample_num = self.mask.shape[1]
 
    def reconstruct_ref_pts(self, bridge_camera):
        self.ref_re_cameras = np.zeros((3, self.sample_num))        
        
        params_db = camerasParamsDatabase()
        for camera in self.camera_list:
            c_idx = camera_list.index(camera) #camera index
            if bridge_camera == camera:
                key = bridge_camera + "_params_world_csys"
            else: 
                key = bridge_camera + "_"+ camera + "_params_world_csys"
                
            proj = params_db.get_data(key)
            
            if proj:
                proj = np.array(proj).reshape((3,4)) 
                self.proj_cameras_M[c_idx, :, :] = proj    
        self.proj_cameras_list = \
            self.proj_cameras_M.reshape((1,12*self.cameras_num)).tolist()[0]
        for p_idx in np.arange(self.sample_num):
            X = []
            P = []
            c_list = []            
            for c_idx in np.arange(self.cameras_num):
                if self.mask[c_idx, p_idx] == 1:
                    X.append(self.img_filter_cameras[c_idx,:,p_idx])
                    P.append(self.proj_cameras_M[c_idx, :, :])
                    c_list.append(c_idx)
            if len(X)>1:      
                ref_re_pts = triangulate_multi_cameras(X, P)    
                self.ref_re_cameras[:,p_idx] = ref_re_pts   
                #ref_pts = self.ref_filter_cameras[c_idx,:,p_idx]   
                #print("Camera", c_list)
                #print(np.abs(ref_re_pts - ref_pts))
        self.ref_re_cameras_list = \
            self.ref_re_cameras.reshape((1,3*self.sample_num)).tolist()[0] 
            
    def get_observations_num(self):
        self.observations_num = (self.mask>0).sum()
                
    def get_reconstruct_error(self, pts_num = None, is_plot=False):
        if not pts_num:
            pts_num = self.sample_num
        self.error_ref_pts = np.zeros((3, pts_num))
        dist_error_pts = np.zeros((3, int(pts_num/3)))         
        for p_idx in np.arange(pts_num):
            for c_idx in np.arange(self.cameras_num):
                if self.mask[c_idx, p_idx] == 1:
                    ref_pts = self.ref_filter_cameras[c_idx,:,p_idx]
                    break
            ref_re_pts = self.ref_re_cameras[:,p_idx]
            if p_idx%3==0:
                counter_3 = 0
                red_res = ref_re_pts
            elif p_idx%3==1 and counter_3 ==0:
                counter_3 += 1
                blue_res = ref_re_pts 
            elif p_idx%3==2 and counter_3 ==1:
                counter_3 += 1
                green_res = ref_re_pts     
            if counter_3 ==2:
                len_RtoG = abs(np.linalg.norm(red_res- green_res) -0.6)
                len_RtoB = abs(np.linalg.norm(red_res- blue_res) -0.4)           
                len_GtoB = abs(np.linalg.norm(green_res- blue_res) -1.0)
                dist_error_pts[:, int(p_idx/3)] = np.array([len_RtoG, len_RtoB, len_GtoB])
                counter_3 = 0           
            self.error_ref_pts[:, p_idx] = np.abs(ref_re_pts - ref_pts)
        if is_plot:
 
            error_plot(dist_error_pts, titles = ["GreenToRed", "RedToBlue", "GreenToBlue"])        
            error_plot(self.error_ref_pts)   
                 
    def cal_cameras_graph(self):
        camera_groups = tuple(itertools.combinations(self.camera_list,2)) 
        for group in camera_groups:
            c1_idx = camera_list.index(group[0])
            c2_idx = camera_list.index(group[1])   
            common_pts_m = np.row_stack((self.mask[c1_idx,:], self.mask[c2_idx,:]))
            common_pts_num = len(np.where((common_pts_m[0,:] == 1) & (common_pts_m[1,:]==1))[0])
           # print("camera%d camera%d common pts %d"%(c1_idx, c2_idx, common_pts_num))
            weight = 1/common_pts_num
            self.cameras_graph.add_edge(group[0], group[1], weight)            
    
    def get_best_group(self, camera1):
        camera2 = self.cameras_graph.get_border_cost(camera1)[0][0]
        return(camera1, camera2)

def get_op_params(camera1, camera2, camera_list, multi_c_p, params_db, is_get_f_from_db = False):
    """
    get optimize params, 
    R_T: camera2 coordinate system to camera1 coordinate system 
    """ 
    c1_idx, c2_idx = camera_list.index(camera1), camera_list.index(camera2)

    common_pts_m = np.row_stack((multi_c_p.mask[c1_idx,:], multi_c_p.mask[c2_idx,:]))
    common_pts = np.where((common_pts_m[0,:] == 1) & (common_pts_m[1,:]==1))[0]
    img_data1 = multi_c_p.img_filter_cameras[c1_idx,:,common_pts].T
    img_data2 = multi_c_p.img_filter_cameras[c2_idx,:,common_pts].T 
    if is_get_f_from_db:
        f0 = params_db.get_data(camera1 +"_int") 
        f1 = params_db.get_data(camera2 +"_int") 
        if f0 and f1:
            op_x = get_params_from_stereo_vision(img_data1, img_data2, \
                                                 (f0, f1))  
        else:
            op_x = get_params_from_stereo_vision(img_data1,img_data2)
            params_db.store_data(camera2 +"_int", op_x[1]) 
 #           print("store int params:", camera2, op_x[1])
    else:
        op_x = get_params_from_stereo_vision(img_data1,img_data2)
        params_db.store_data(camera2 +"_int", op_x[1]) 
#        print("store int params:", camera2, op_x[1])    
    f0 = op_x[0]
    f1 = op_x[1]
    R_M = Rodrigues(np.array(op_x[2:5]))[0]
    t = np.array(np.array(op_x[5:]))
    R_T = np.column_stack((R_M, t)) 
    
    return op_x, f0, f1, R_T 


            
def get_params_from_muti_cameras(data_processor, bridge_camera, camera_list):
    """
    multiple cameras calibration.
    1. get the focal length based on the 
    """
    t0 = time.time() 
    params_db = camerasParamsDatabase()
    if bridge_camera in camera_list:
        bridge_camera_idx = camera_list.index(bridge_camera)
        other_cameras = camera_list[:bridge_camera_idx] + camera_list[bridge_camera_idx+1:]
    else:
        other_cameras = camera_list
    p = data_processor
#    for camera in camera_list:
    for camera in [bridge_camera]: 
        best_group = list(p.get_best_group(camera))
        
        if not params_db.get_data( camera +"_int"):
            if bridge_camera in best_group:
                b_idx = best_group.index(bridge_camera)
                best_group = best_group[b_idx], best_group[1-b_idx]
            print("best_group", best_group)
            camera1, camera2 = best_group
            key = camera1+"_"+camera2
            print("Int param key:", key)
            if not params_db.get_data(key):
                op_x, f0, f1, R_T = get_op_params(camera1, camera2, camera_list, p, params_db)
                if camera == camera1 :
                    params_db.store_data(camera +"_int", f0) 
                else:
                    params_db.store_data(camera +"_int", f1) 
                params_db.store_data(key, op_x)           

    for camera in other_cameras:
        Best_RT = np.eye(4, 4)
        key = bridge_camera+"_"+camera
        key_inv = camera+"_"+bridge_camera
        path = p.cameras_graph.dijkstra(camera, bridge_camera)[1]   
        RT_list = []
        groups = gen_group(path, 2)
        print("Ext groups:", groups)

        if (get_ext_params_from_db(bridge_camera, camera, params_db) is None):  
            for group in groups:
                group_key = group[1] + "_" + group[0]
                if not params_db.get_data(group_key):
                    op_x, f0, f1, R_T = get_op_params(group[1], group[0],\
                                                      camera_list, p, params_db, True)
                    
                    params_db.store_data(group_key, op_x)  
                else:
                    op_x = params_db.get_data(group_key)
                    R_M = Rodrigues(np.array(op_x[2:5]))[0]
                    t = np.array(np.array(op_x[5:]))
                    R_T = np.column_stack((R_M, t))                 
                RT_list.append(np.row_stack((R_T, np.array([0,0,0,1]))))    
            for R_T in RT_list:
                Best_RT = Best_RT.dot(R_T)   
            Best_RT_inv = inv(Best_RT)[0:3,:]
            Best_RT = Best_RT[0:3,:]
            params_db.store_data(key+"_ext_camera_csys", Best_RT.tolist()) 
            params_db.store_data(key_inv+"_ext_camera_csys", Best_RT_inv.tolist())                            
    t1 = time.time()
    print("Mutiple calibrate take {0:.0f} seconds".format(t1 - t0))    

def verify_BA(bridge_camera, camera_list, img_filter_cameras, ref_filter_cameras,\
              ref_re_cameras, mask, pts_num):
    """
    Verify the calibration effect after bundle adjustment
    """
    error_ref_pts = np.zeros((3, pts_num)) 
    dist_error_pts = np.zeros((3, int(pts_num/3))) 
    cameras_num = len(camera_list)
    proj_cameras_M = np.zeros((cameras_num, 3, 4))
    params_BA = camerasParamsDatabase(bridge_camera +"_Params_BA.db")
    for camera in camera_list:
        c_idx = camera_list.index(camera) #camera index

        proj = params_BA.get_data(camera)
        if proj:
            proj = np.array(proj).reshape((3,4)) 
            proj_cameras_M[c_idx, :, :] = proj   

            
    for p_idx in np.arange(pts_num):
        X = []
        P = []
        
        for c_idx in np.arange(cameras_num):
            if mask[c_idx, p_idx] == 1:
                last_c_idx = c_idx
                X.append(img_filter_cameras[c_idx,:,p_idx])
                P.append(proj_cameras_M[c_idx, :, :])
        if len(X)>1:      
            ref_re_pts_BA = triangulate_multi_cameras(X, P)               
            ref_pts = ref_filter_cameras[last_c_idx,:,p_idx]
            error_ref_pts[:, p_idx] = np.abs(ref_re_pts_BA - ref_pts)

        if p_idx%3==0:
            counter_3 = 0
            red_res = ref_re_pts_BA
        elif p_idx%3==1 and counter_3 ==0:
            counter_3 += 1
            blue_res = ref_re_pts_BA 
        elif p_idx%3==2 and counter_3 ==1:
            counter_3 += 1
            green_res = ref_re_pts_BA     
        if counter_3 ==2:
            len_RtoG = abs(np.linalg.norm(red_res- green_res) - len_relation["Red_Green"])
            len_RtoB = abs(np.linalg.norm(red_res- blue_res) - len_relation["Red_Blue"])         
            len_GtoB = abs(np.linalg.norm(green_res- blue_res) -len_relation["Blue_Green"])

            dist_error_pts[:, int(p_idx/3)] = np.array([len_RtoG, len_RtoB, len_GtoB])
            counter_3 = 0
            
    error_plot(dist_error_pts, titles = ["GreenToRed", "RedToBlue", "GreenToBlue"])        
            
    error_plot(error_ref_pts)
    print("<Mean> Fig1:%f  Fig2:%f  Fig3:%f"% \
          (np.mean(error_ref_pts[0,:]),np.mean(error_ref_pts[1,:]),np.mean(error_ref_pts[2,:])))       
    print("<Std> Fig1:%f  Fig2:%f  Fig3:%f"% \
          (np.std(error_ref_pts[0,:]),np.std(error_ref_pts[1,:]),np.std(error_ref_pts[2,:])))       
    result = [np.mean(error_ref_pts[0,:]),\
            np.mean(error_ref_pts[1,:]),\
            np.mean(error_ref_pts[2,:]),\
            np.std(error_ref_pts[0,:]),\
            np.std(error_ref_pts[1,:]),\
            np.std(error_ref_pts[2,:])]   
    return result



def error_plot(error_data, titles = ["X axis", "Y axis", "Z axis"]):
    pts_num = error_data.shape[1]
    x = np.arange(0, pts_num, 1)
    
    # red dashes, blue squares and green triangles
    plt.figure(1)
    plt.plot(x, error_data[0,:], 'r*')
    plt.title(titles[0])
    plt.figure(2)
    plt.plot(x, error_data[1,:], 'bo')

    plt.title(titles[1])
    plt.figure(3)
    plt.plot(x, error_data[2,:], 'g.')
    plt.title(titles[2])
    plt.show()    


def bundle_adjustment(bridge_camera, camera_list, pts_num =None, is_restart=False): 
    p = multi_cameras_data_processor(camera_list, pts_num)
    p.reconstruct_ref_pts(bridge_camera)      
    init_params = p.proj_cameras_list + p.ref_re_cameras_list
    bundle_adjust(init_params, bridge_camera, p.mask,p.img_filter_cameras, \
                  p.sample_num, p.cameras_num, p.camera_list, is_restart=is_restart)

def verify_params(bridge_camera, camera_list):
    p = multi_cameras_data_processor(camera_list)
#    p.reconstruct_ref_pts(bridge_camera)  
#    p.get_reconstruct_error(is_plot=True)
    
    result = verify_BA(bridge_camera, camera_list, p.img_filter_cameras, \
                       p.ref_filter_cameras, p.ref_re_cameras, p.mask, p.sample_num)
    return result

def gen_group(lst, n):
    groups = []
    for i in range(0, len(lst), n-1):
        val = lst[i:i+n]
        if len(val) == n:
          groups.append(tuple(val))
    return groups


def multi_calib(data_p, bridge_camera = None, is_restart=True):
    if not bridge_camera:
        cost_of_cameras = data_p.cameras_graph.get_cost_of_nodes()
        bridge_camera = cost_of_cameras[0][0]
    print("bridge_camera", bridge_camera)
    
    if is_restart:
        params_BA_db = camerasParamsDatabase(bridge_camera +"_Params_BA.db") 
        params_BA_db.del_all()
        params_db = camerasParamsDatabase()
        params_db.del_all()
    get_params_from_muti_cameras(data_p, bridge_camera, camera_list)
    level_calibrate(bridge_camera, data_p, data_p.img_accuracy)     
    bundle_adjustment(bridge_camera, camera_list, is_restart=True)
    error_result = verify_params(bridge_camera, camera_list)
    return error_result

def test(excel_path):
    """
    img_accuracy: ndigits precision after the decimal point
    """
    img_accuracy_l = [1,2,3,4]
    #img_accuracy_l = [1]
    error_data_l = []
    for img_accuracy in img_accuracy_l:
        data_p = multi_cameras_data_processor(camera_list, \
                                              img_accuracy = img_accuracy)
#        cost_of_cameras = data_p.cameras_graph.get_cost_of_nodes()
        cost_of_cameras = data_p.cameras_graph.get_cost_with_direct_borders()

        error_data = {}
        for camera in cost_of_cameras[0]:
            error_data[camera] = multi_calib(data_p, camera,is_restart = True)
        error_data_l.append(error_data)
    generate_report(['.'+'x'*m for m in img_accuracy_l], cost_of_cameras[1], error_data_l,excel_path)

test(r'F:\Project\stereo vision\demo\report.xls')


#verify_params("camera8", camera_list)
