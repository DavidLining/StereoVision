# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 11:36:25 2018

@author: Morgan.Li
"""
from calibration.calibrate import CameraCalibDataCollector, camera_database
from reconstruction.position import stereo_position_func, optimize_position_func
import os,getopt,sys
from db_lib.database import CameraCalDatabase
from calibration.level_calib import level_calib_data_collect

def main():
    point_flag = None
    camera1_flag = None
    camera2_flag = None
    eight_camera_coordinates_file_path = camera_database.coordinate_file
    camera_list = camera_database.camera_list
    recalib_flag = False
    is_sync = True
    try:
        opts, args = getopt.getopt(sys.argv[1:], "ahpcld:f:", ["auto", "help", "position", "collect", "del=", "file"])
    except getopt.GetoptError as err:
        print(err)
        sys.exit()
    for op, value in opts:
        if op in ("-a", "--auto"):
            if args[0] in camera_database.point_list:
                if len(args)>1:
                    if(recalib_flag=="True"):
                        recalib_flag = True
                    else:
                        recalib_flag = False
                optimize_position_func(args[0], recalib_flag)
            else:
                print("Point flag must in ['Red', 'Blue', 'Green', 'Yellow']!")
        elif op in ("-p", "--position"):
            arg_list = args
            camera1_flag = arg_list[0]
            camera2_flag = arg_list[1]
            point_flag = arg_list[2]
            if(len(arg_list)>3):
                recalib_flag = arg_list[3]
                if(recalib_flag=="True"):
                    recalib_flag = True
                else:
                    recalib_flag = False
            if(len(arg_list)>4):
                is_sync = arg_list[4]
                if(is_sync=="True"):
                    is_sync = True
                else:
                    is_sync = False
            stereo_position_func(camera1_flag,camera2_flag, point_flag, recalib_flag, True, is_sync)
            sys.exit()
        elif op in ("-h", "--help"):
            print( "Usage main.exe to imitate Visual Positioning with two camera.\r\n \r\n", \
                  "-h --help: help information. \r\n \r\n" ,                \
                  "-p, --position <CAMERA1> <CMAERA2> <POINT> [RECALIB_FLAG]:\r\n"  \
                  " CMAERA1, CMAERA2: 'camera1'~'camera8', select required cameras to do positioning", \
                  " POINT: 'Blue', 'Green', 'Yellow', 'Red', specified point on target",\
                  " RECALIB_FLAG: 'True' or 'False', use True to recalibrate specified camera.\r\n \r\n", \
                  " Example: '-p camera1 camera8 Blue',  '-p camera1 camera8 Blue True'\r\n",\
                  "-c, --collect [CMAERAx] <CAMERAx>: \r\n", \
                  " CAMERAx: 'camera1'~'camera8', collect the initialized data used for camera calibration, you can use 'all' to collect data for all cameras \r\n \r\n", \
                  " Example: '-c camera1 camera2 camera3', '-c all' \r\n",\
                  "-d, --del [SOURCE] [CAMERA] <CAMERAx>: \r\n", \
                  " SOURCE: 'file', 'db', delele initialized data files or database", \
                  " CMAERA, CAMERAx: 'camera1'~'camera8', delete the initialized data file or database(storing calibration matrix), you can use 'all' to delete data for all cameras \r\n \r\n", \
                  " Example: '-d file camera1 camera2', '-d db all' \r\n",\
                  "-f, --file [COORDINATE_FILE_PATH]: \r\n", \
                  " COORDINATE_FILE_PATH: set file path to store coordinate data \r\n"\
                )
            sys.exit()
        elif op in ("-l"):
            level_calib_data_collect(camera_list)
        elif op in ("-c", "--collect"):
            keep_reso = True
            arg_list = args
            if(len(arg_list)>0):
                keep_reso = arg_list[0]
                if(keep_reso=="False"):
                    keep_reso = False
                else:
                    keep_reso = True
            counter = 0
            while(counter<20000):
                try:
                    counter +=1
                    data_collector = CameraCalibDataCollector(keep_reso)
                    data_collector.data_collect(eight_camera_coordinates_file_path, camera_list)
                    print("Collect data counter:%d. \r\n"%(counter))
                except PermissionError:
                    pass

            print("Collect data end!")
            sys.exit()
        elif op in ("-d", "--del"):
            arg_list = args
            if value=="all":

                data_collector = CameraCalibDataCollector()
                data_collector.clear(camera_list)
                print("Succeed to clear data from: ", camera_list)
                camera_database.del_all()
                print("Clear database!")

            sys.exit()
        elif op in ("-f", "--file"):
            if os.path.exists(value):
                eight_camera_coordinates_file_path = value
                camera_database.set_coordinate_file_path(eight_camera_coordinates_file_path)
            else:
                print("Invalid coordinates file!")
            sys.exit()
        else:
            print("Unhandled option")
    sys.exit()


main()