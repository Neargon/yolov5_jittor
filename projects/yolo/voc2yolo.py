# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import random
import shutil
classes=["mask", 'nomask']  # class names


def clear_hidden_files(path):
    dir_list = os.listdir(path)
    for i in dir_list:
        abspath = os.path.join(os.path.abspath(path), i)
        if os.path.isfile(abspath):
            if i.startswith("._"):
                os.remove(abspath)
        else:
            clear_hidden_files(abspath)

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(path,index):
    # in_file = open(path).read().decode("gb18030","ignore")
    if index == 0:
        out_file = open(path.replace("\\label\\","\\labels\\test\\").replace(".xml",".txt"),"w")
    elif index == 1:
        out_file = open(path.replace("\\label\\","\\labels\\val\\").replace(".xml",".txt"), "w")
    else:
        out_file = open(path.replace("\\label\\","\\labels\\train\\").replace(".xml",".txt"), "w")
    # with open(path, 'r', encoding='gbk') as f:
    #     xml_content = f.read()

    # 使用 ET.fromstring 解析解码后的 XML 内容
    # root = ET.fromstring(xml_content)
    tree=ET.parse(path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        try:
            difficult = obj.find('difficult').text
        except:
            difficult = 0 

            # print(path)
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    
    out_file.close()
path = "C:\\Users\\Admin\\Desktop\\yolox\\dataset\\label"
a = 0
fte = open(os.path.join(path.replace("label","labels"),"test.txt"),"w")
ftr = open(os.path.join(path.replace("label","labels"),"train.txt"),"w")
fva = open(os.path.join(path.replace("label","labels"),"val.txt"),"w")
for xml_name in os.listdir(path):

    if ".xml" in xml_name:
        xml_path = os.path.join(path,xml_name)
        pic_Path = xml_path.replace("\\label\\","\\data\\").replace(".xml",".jpg")
        if os.path.exists(pic_Path):
            a = a + 1
            if a % 10 == 0:
                
                convert_annotation(xml_path, 0)
                source_path =pic_Path.replace("\\data\\","\\images\\test\\")
                shutil.copy(pic_Path,source_path)
                fte.write(source_path+"\n")
            elif a % 10 == 1:
                convert_annotation(xml_path, 1)
                source_path =pic_Path.replace("\\data\\","\\images\\val\\")
                shutil.copy(pic_Path,source_path)
                fva.write(source_path+"\n")
            else:
                convert_annotation(xml_path, 2)
                source_path =pic_Path.replace("\\data\\","\\images\\train\\")
                shutil.copy(pic_Path,source_path)
                ftr.write(source_path+"\n")
