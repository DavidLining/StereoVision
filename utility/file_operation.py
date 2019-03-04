# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 17:39:49 2018

@author: Morgan.Li
"""
import os
import xlwt


def walk_all_files(rootdir,query):  
    for parent,dirnames,filenames in os.walk(rootdir):   #for循环自动完成递归枚举  #三个参数：分别返回1.父目录（当前路径） 2.所有文件夹名字（不含路径） 3.所有文件名字  
        for dirname in dirnames:                         #输出文件夹信息  
            pass  
  
        for filename in filenames:                       #输出文件信息  
           if is_filename_contain_word(filename,query):  
                return os.path.join(parent,filename)
    return None
def is_filename_contain_word(filename,query_word):  
    if query_word in filename:
        return True
    else:
        return False
    
 
def store_result(data):
    result_file_path = os.path.join(os.getcwd(), "position_result.txt")    
    result_file = open(result_file_path, 'a')
    result_file.truncate()
    result_file.write(data)
    result_file.close()



def set_style(name, height, bold = False):  
    style = xlwt.XFStyle()   #初始化样式  
      
    font = xlwt.Font()       #为样式创建字体  
    font.name = name  
    font.bold = bold  
    font.color_index = 4  
    font.height = height  
      
    style.font = font  
    return style  
  

def generate_report(img_accuracy_l, cost_data, error_data_l,excel):
    """
    generate the report about reconstruct accuracy
    """
    book = xlwt.Workbook()
    for img_accuracy in img_accuracy_l:
        error_data = error_data_l[img_accuracy_l.index(img_accuracy)]
        data_sheet = book.add_sheet(img_accuracy)
        row0 = [u'#Bridge Camera', u'Cost', u'Img Data Accuracy',\
                u'X(Mean)', u'Y(Mean)', u'Z(Mean)', \
                u'X(Std)', u'Y(Std)', u'Z(Std)']
        for j in range(len(row0)):
            data_sheet.write(0, j, row0[j], set_style('Times New Roman', 220, True))
        row_num = 0
        for key in error_data.keys():
            bridge_camera = key
            cost = cost_data[key]
            error = error_data[key]
    
            row = ['#'+bridge_camera, cost, img_accuracy]
            row.extend(error)
            row_num = row_num + 1
            for j in range(len(row)):
                data_sheet.write(row_num, j, row[j], set_style('Times New Roman', 220, True))
    
    book.save(excel)




