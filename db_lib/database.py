# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 16:57:41 2018

@author: Morgan.Li
"""

import pickledb, os
import numpy as np


db_dir =  os.path.join(os.getcwd(), "db")
if not os.path.exists(db_dir):
    os.makedirs(db_dir)
db_dir = "F:\Project\stereo vision\demo\EightCameraDemo\db"


class CameraParamDatabase(object):

    def __init__(self, db_name='cameraParam.db'):
        db_name = os.path.join(db_dir, db_name)
        self.db = pickledb.load(db_name, False)
        self.db_keys = self.db.getall()
        #self.coordinate_file = walk_all_files(os.path.dirname(os.getcwd()), "Coordinates.txt")
        self.coordinate_file = None
        self.last_coordinate_file = None
        self.sencond_last_coordinate_file = None
        self._get_coordinate_file_path()
        self.camera_list = ['camera1', 'camera2', 'camera3', 'camera4', \
                            'camera5', 'camera6', 'camera7', 'camera8'] 
        self.point_list = ['Red', 'Blue', 'Green', 'Yellow']
        self.init_coordinate = {'Red':[-0.5, 0.125, 0.25], 'Green':[0.5, 0.125, -0.25], \
                                'Yellow':[0.5, 0.125, 0.25], 'Blue':[-0.5, 0.125, -0.25]}

    def search(self,camera):
        proj_mat = None
        if camera in self.db_keys:
            proj_mat = self.db.get(camera)
        else:
            self.db.set(camera, None)
            self.db.dump() 
        if proj_mat:
            np.array(proj_mat).reshape((3,4))
            
        return proj_mat
    
    
    def set(self, camera, proj_mat):
        self.db.set(camera, proj_mat.tolist())
        self.db.dump() 
    
    def _get_coordinate_file_path(self):
        default_coordinate_file = r'F:\Project\stereo vision\demo\EightCameraDemo\Coordinates.txt'
        if "coordinates_file" in self.db_keys:
            self.coordinate_file = self.db.get("coordinates_file") 
        else:
            if not self.coordinate_file:
                self.coordinate_file = default_coordinate_file
            self.db.set("coordinates_file", self.coordinate_file)
            self.db.dump()
                      
        self.last_coordinate_file = os.path.join(os.path.dirname(self.coordinate_file), "Coordinates_last.txt")
        self.sencond_last_coordinate_file =  os.path.join(os.path.dirname(self.coordinate_file), "Coordinates_sec_last.txt")   
    
    def set_coordinate_file_path(self, file_path):
        self.db.set("coordinates_file", file_path)
        self.db.dump()
        
        
    def delete(self, key):
        if key in self.db_keys:
            self.db.rem(key)
            
            
    def del_all(self):
        self.db.deldb()
        self.db.dump()
        

class db_base_class(object):
    def __init__(self, dbname):
        db = os.path.join(db_dir, dbname)
        try:
            self.db = pickledb.load(db, False)  
        except:
            os.remove(db)
            self.db = pickledb.load(db, False)  
        
    def store_data(self, key, data):
        self.db.set(key, data)
        self.db.dump()        

    def get_data(self, key):
        data = self.db.get(key)                       
        return data        
    
    def delete(self, key):
        if key in self.db.getall():
            self.db.rem(key)
                   
    def del_all(self):
        self.db.deldb()
        self.db.dump()  


class camerasParamsDatabase(db_base_class):
    def __init__(self, name="Cameras_Params.db"):
        self.db_path = os.path.join(db_dir, name)

        self.db = pickledb.load(self.db_path, False)  
    
    def is_db_null(self):
        if not os.path.exists(self.db_path):
            return True
        size = os.path.getsize(self.db_path)
        if size > 1024:
            # greater than 1Kb
            return False
        else:
            return True
        
class LevelCalibDB(db_base_class):
    def __init__(self, name="level_calib.db"):
        db = os.path.join(db_dir, name)

        self.db = pickledb.load(db, False)  


class camerasGraphDB(db_base_class):
    def __init__(self, name="cameras_graph.db"):
        db = os.path.join(db_dir, name)

        self.db = pickledb.load(db, False)  
        
        
class CameraCalDatabase(db_base_class):
    def __init__(self, camera):
        self.camera = camera
        db_name = camera + "Calib.db"
        db_name = os.path.join(db_dir, db_name)
        self.db = pickledb.load(db_name, False)

    
    def get_cal_data(self, key):
        cal_data = self.db.get(key)            
            
        return cal_data
 
    def store_cal_data(self, cal_data):
        key = str(len(self.db.getall())+1)
        self.store_data(key, cal_data)
            
        