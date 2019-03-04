# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 09:34:49 2018

@author: Morgan.Li
"""
"""
Classes for calibrating homemade stereo cameras.
"""

import os
import numpy as np
from utility.file_operation import walk_all_files
from utility.math_operation import accuracy_process
from db_lib.database import CameraParamDatabase, CameraCalDatabase, LevelCalibDB
from calibration.internal_params import triangulate_point

camera_database = CameraParamDatabase()

class CameraCalibrator(object):

    """
    A stereo camera calibration.
    Get the projection matrix of specified camera
    """
    def __init__(self, camera, input_folder=None, pts3d_filename=None, \
                 pts2d_filename=None, recalib=False, keep_reso = False):
        """
        Initialize camera calibration.
        keep_reso: True to keep current resolution.
        """
        
        #: Projection matrices (3x4 projection matrix) 
        self.proj_mat = None
        self.points_3d_array = []
        self.points_2d_array = []
        self.camera = camera
        self.recalib = recalib
        self.database = camera_database
        self.keep_reso = keep_reso
        input_folder = os.path.join(input_folder, self.camera)
        self.search()
        if ((self.proj_mat==None) and input_folder and pts3d_filename and pts2d_filename) or self.recalib:
            print("Recalibrate")
            self.load(input_folder,pts3d_filename, pts2d_filename)
            self.camera_calibrate(self.points_3d_array, self.points_2d_array) 
    
    def search(self):
        """
        search camera calibrate information in database
        """
        self.proj_mat = self.database.search(self.camera)
    
    def load(self, input_folder, pts3d_filename, pts2d_filename):
        """
        load file form folder and parse it 
        """
        points_3d_file_path = walk_all_files(input_folder, pts3d_filename)
        points_2d_file_path = walk_all_files(input_folder, pts2d_filename)
        if(points_3d_file_path and points_2d_file_path):
            
            points_3d_file = open(points_3d_file_path, 'r', encoding='utf-8')
            points_3d_file.seek(0, 0)
            points_2d_file = open(points_2d_file_path, 'r', encoding='utf-8')
            points_2d_file.seek(0, 0)

            for line in points_3d_file:
                line = line.strip().strip('\n')
                if line != '':
                    line_s = line.strip('(').strip(')').split(',')
                    for i in range(3):
                        self.points_3d_array.append(float(line_s[i].strip()))
                
            for line in points_2d_file:
                line = line.strip().strip('\n')
                if line != '':
                    line_s = line.strip('(').strip(')').split(',')
                    for i in range(2):
                        point_2d_value = float(line_s[i].strip())   
                        self.points_2d_array.append(point_2d_value)
            self.points_3d_array = np.array(self.points_3d_array).reshape((int(len(self.points_3d_array)/3),3))
            self.points_2d_array = np.array(self.points_2d_array).reshape((int(len(self.points_2d_array)/2),2))
        else:
            raise Exception("Error: data file is not exist.")             

    def export(self, output_folder):
        """Export matrices as ``*.npy`` files to an output folder."""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            

    def camera_calibrate(self, points_3d, points_2d):
        """
        camera calibrate, get the projection matrix
        """
        #" points_3d: array, points_2d: array "
        if(len(points_3d)>5 and len(points_2d)>5 and len(points_3d) == len(points_2d)):
            #"refer to paper http://www.ixueshu.com/document/109b65925f1192f1318947a18e7f9386.html"
            ones_column = np.ones(len(points_3d))
            points_3d_h = np.column_stack((points_3d, ones_column))
            
            K = np.zeros((2*len(points_3d), 11))
            #"U vector"
            U = np.zeros((2*len(points_3d), 1)) 
            for i in range(len(points_3d)):
                K[2*i] =  np.c_[np.matrix(points_3d_h[i]), \
                         np.matrix(np.zeros((1,4))), np.matrix(-points_2d[i][0]*points_3d_h[i][:3])]
                K[2*i+1] =  np.c_[np.matrix(np.zeros((1,4))), \
                         np.matrix(points_3d_h[i]), np.matrix(-points_2d[i][1]*points_3d_h[i][:3])]
                U[2*i] =  points_2d[i][0]
                U[2*i+1] = points_2d[i][1]
            K_t = np.transpose(K)
            K_s = np.dot(K_t, K)
            K_inv = np.dot(np.matrix(K_s).I, np.matrix(K_t))
            #M vector, not include m34
            M_v = np.dot(K_inv, U) 
            #M matrix
            self.proj_mat = np.row_stack((np.matrix(M_v), np.matrix([1.0])))
            self.proj_mat = np.array(self.proj_mat).reshape((3,4))
            
            self.database.set(self.camera, self.proj_mat)
            #print(self.proj_mat)
        else:
            raise Exception("Number of points is not much enough.")
            
class StereoCalibrator(object):

    """A class that calibrates stereo cameras by finding chessboard corners."""

    def __init__(self, proj_1, proj_2, keep_reso = False):
        #: Camera matrices (M)
        self.cam_mats = {"1": None, "2": None}
        #: extrinsic parameters
        self.extr_mats = {"1": None, "2": None}   
        # projection matrixs
        self.proj_mats = {"1": proj_1, "2": proj_2} 
        
        self.keep_reso = keep_reso
    
    def load(self, file_path, camera1, camera2):
        """
        load data from 'file_path', 
        'camera1' and 'camera2' represent selected cameras
        """
        data_collector = CameraCalibDataCollector(self.keep_reso)
        
        return data_collector.load(file_path, camera1), data_collector.load(file_path, camera2)
            
    def calibrate(self, x1, x2, proj_1 = None, proj_2 = None):
        """
        two-dimensional coordinate conversion
        return three-dimensional coordinate value
        """
        
        resturct_3d = triangulate_point(x1, x2, proj_1, proj_2)        
        return resturct_3d


class CameraCalibDataCollector(object):
    """
    collect required data that used for camera calibrate 
    """
    def __init__(self, keep_reso=False):
        self.keep_reso = keep_reso
    
    def _parse_data(self, init_data):
        ref_csys = {"Red": None, "Blue": None, "Yellow": None, "Green": None}
        pic_csys = {"Red": None, "Blue": None, "Yellow": None, "Green": None}
        line_size = len(init_data)
        for i in range(line_size):
            line = init_data[i].strip('\n')  
            if "2D" in line:
                pic_info =  init_data[i+1:i+5]
                pic_csys = self._get_coordinate(pic_info, "2D")
            if "3D" in line:
                ref_info = init_data[i+1:i+5]
                ref_csys = self._get_coordinate(ref_info, "3D")  
        return pic_csys, ref_csys
                
    def _get_coordinate(self, init_data, dimension = "3D"):
        coordinate_info = {"Red": None, "Blue": None, "Yellow": None, "Green": None}
        for line in init_data:
            line = line.strip('\n')
            for flag in coordinate_info.keys():
                if flag in line:
                    line = line.strip(flag + ":")
                    line = line.strip("(").strip(")").split(",")                   
                    coor_v = [float(x) for x in line]
                    #if last item in coor_v equal to zero means that 
                    #the value exceed the image limit
                    if(dimension == "2D"):
                        if (coor_v[2] != 0):
                            coor_v = None   
                        else:
                            if not self.keep_reso:
                                coor_v = [accuracy_process(x,self.keep_reso) for x in coor_v]
                            #set last item equal to 1 to meet homogeneous equation
                            coor_v[2] = 1.0
                    if coor_v:
                        coordinate_info[flag] = coor_v
        return coordinate_info  
     
    def clear(self, camera_list):
        for camera in camera_list:
            db = CameraCalDatabase(camera)
            db.del_all()
    
    def load(self, file_path, camera_list):
        """
        load data from 'file_path', 
        'camera' represent selected cameras
        """
        cameras_data = {}
        data_file = open(file_path, 'r', encoding='utf-8')
        data_file.seek(0, 0) 
        lines = data_file.readlines();
        data_file.close()
        line_num = len(lines);
        for i in range(line_num):
            line = lines[i].strip('\n')  
            for camera in camera_list:
                if camera in line:
                    camera_data = {"3d": None, "2d": None}
                    init_data = lines[i+2:i+14]
                    camera_data["2d"], camera_data["3d"] = self._parse_data(init_data)
                    cameras_data[camera] = camera_data
                    break
        return  cameras_data    

    """
    collect camera initial calibrate data from file_path
    """    
    def data_collect(self, file_path, camera_list, is_level=False):
        cameras_data = self.load(file_path, camera_list)
        self._data_store(camera_list, cameras_data, is_level) 
        
    def _data_store(self, camera_list, data_source_list, is_level=False):
        if is_level:
            level_data_db = LevelCalibDB()
        for camera in camera_list:
            data_source = data_source_list[camera]
            del_keys = []
            for key in data_source["2d"].keys():
                if not data_source["2d"][key]:
                    del_keys.append(key)
            for key in del_keys:
                data_source["2d"].pop(key)
                data_source["3d"].pop(key)
            if not is_level:               
                db = CameraCalDatabase(camera)
                db.store_cal_data(data_source)
            else:
                level_data_db.store_data(camera, data_source)
        