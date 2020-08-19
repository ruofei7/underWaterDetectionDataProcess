
# coding: utf-8

# In[39]:


#将比赛给的XML数据转化为COCO数据的标注格式
import os
import cv2
import json
import xml.dom.minidom
import xml.etree.ElementTree as ET

data_dir = "D:/data/val" #根目录文件，其中包含image文件夹和box文件夹（根据自己的情况修改这个路径）

image_file_dir = os.path.join(data_dir, 'image')
xml_file_dir = os.path.join(data_dir, 'box')

annotations_info = {'images': [], 'annotations': [], 'categories': []}

categories_map = {'holothurian': 1, 'echinus': 2, 'scallop': 3, 'starfish': 4}

for key in categories_map:
    categoriy_info = {"id":categories_map[key], "name":key}
    annotations_info['categories'].append(categoriy_info)

file_names = [image_file_name.split('.')[0]
              for image_file_name in os.listdir(image_file_dir)]
ann_id = 1
for i, file_name in enumerate(file_names):

    image_file_name = file_name + '.jpg'
    xml_file_name = file_name + '.xml'
    image_file_path = os.path.join(image_file_dir, image_file_name)
    xml_file_path = os.path.join(xml_file_dir, xml_file_name)

    image_info = dict()
    image = cv2.cvtColor(cv2.imread(image_file_path), cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    image_info = {'file_name': image_file_name, 'id': i+1,
                  'height': height, 'width': width}
    annotations_info['images'].append(image_info)

    DOMTree = xml.dom.minidom.parse(xml_file_path)
    collection = DOMTree.documentElement

    names = collection.getElementsByTagName('name')
    names = [name.firstChild.data for name in names]

    xmins = collection.getElementsByTagName('xmin')
    xmins = [xmin.firstChild.data for xmin in xmins]
    ymins = collection.getElementsByTagName('ymin')
    ymins = [ymin.firstChild.data for ymin in ymins]
    xmaxs = collection.getElementsByTagName('xmax')
    xmaxs = [xmax.firstChild.data for xmax in xmaxs]
    ymaxs = collection.getElementsByTagName('ymax')
    ymaxs = [ymax.firstChild.data for ymax in ymaxs]

    object_num = len(names)

    for j in range(object_num):
        if names[j] in categories_map:
            image_id = i + 1
            x1,y1,x2,y2 = int(xmins[j]),int(ymins[j]),int(xmaxs[j]),int(ymaxs[j])
            x1,y1,x2,y2 = x1 - 1,y1 - 1,x2 - 1,y2 - 1

            if x2 == width:
                x2 -= 1
            if y2 == height:
                y2 -= 1

            x,y = x1,y1
            w,h = x2 - x1 + 1,y2 - y1 + 1
            category_id = categories_map[names[j]]
            area = w * h
            annotation_info = {"id": ann_id, "image_id":image_id, "bbox":[x, y, w, h], "category_id": category_id, "area": area,"iscrowd": 0}
            annotations_info['annotations'].append(annotation_info)
            ann_id += 1

with  open('./annotations_val.json', 'w')  as f:
    json.dump(annotations_info, f, indent=4)

print('---整理后的标注文件---')
print('所有图片的数量：',  len(annotations_info['images']))
print('所有标注的数量：',  len(annotations_info['annotations']))
print('所有类别的数量：',  len(annotations_info['categories']))


# In[42]:


#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# Author: 杜建军
# @Time: 2020/3/11 10:37
# @IDE: PyCharm
import json
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# 取json数据,画扇形统计图，统计每个类别出现的次数
with open("C:/Users/91190/Desktop/annotations_val.json",'r') as fp:
    print(type(fp))
    loaded_json=json.load(fp)
    print(type(loaded_json))
    class_num=[0,0,0,0]
    print(class_num)
    for annotation in loaded_json['annotations']:
        for key,value in annotation.items():
            if(key=='category_id'):
                class_num[value-1]=class_num[value-1]+1
    print(class_num)

labels = ['0#holothurian', '1#echinus', '2#scallop', '3#starfish']
X = class_num
fig = plt.figure()
plt.pie(X, labels=labels, autopct='%1.2f%%')  # 画饼图（数据，数据对应的标签，百分数保留两位小数点）
plt.title("data_val")
plt.savefig("data_val.jpg")
plt.show()


# In[37]:


#划分验证集
#训练集与验证集比例8：2
#划分方法，每10个样本随机取2个作为验证集
import os
import random
path_train = "D:/data/train"#train文件夹的绝对路径，里面包含box与image两个文件夹
path_val = "D:/data/val"#val文件夹绝对路径，下有box与image两个文件夹，需要提前创建好

val = []
for i, box in enumerate(os.listdir(path+"/box"), 1):
    val.append(box[0:-4])#保存10个样本
    if (i%10 == 0):
        val = random.sample(val, 2)#随机生生成2个作为val
        #将box移动到val/box文件夹
        os.rename(path_train+"/box/"+val[0]+".xml", path_val+"/box/"+val[0]+".xml")
        os.rename(path_train+"/box/"+val[1]+".xml", path_val+"/box/"+val[1]+".xml")
        #将image移动到val/image文件夹
        os.rename(path_train+"/image/"+val[0]+".jpg", path_val+"/image/"+val[0]+".jpg")
        os.rename(path_train+"/image/"+val[1]+".jpg", path_val+"/image/"+val[1]+".jpg")
        #重置val
        val = []
    
    


# In[1]:


cat = {v:i for i,v in enumerate([1,2,3])}
print(cat)
cat


# In[ ]:


#推理程序，infer test img，并输出csv文件

import csv
import os
from mmdet.apis import init_detector, inference_detector, show_result
import mmcv

#构建model并读取pretrained
config_file = 'configs/faster_rcnn_r50_fpn_1x.py'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

pathImg =""#test图片文件夹的path 
csvTest = open("")#读取实例csv
read = csv.reader(csvTest)
resultCsv = open("", "a")#打开result csv文件，检测的结果将存放在这里面，需要提前新建好

#取testImg文件夹中的每一张图片
for line in read:
    try:
        img = os.path.join(path, line[1])#img的绝对路径
        #将img传给model，产生输出
        # test a single image
        result = inference_detector(model, img)
        # save the visualization results to image files
        show_result(img, result, model.CLASSES, out_file=os.path.join(path, 'result', line[1]))
        
        #将结果写进csv,result array中保存了很多bbox，一次只写一个
        for i in range(result.shape):
            #产成大于阈值的list并将该list命名为record
            
            
            
            
            writer = csv.writer(resultCsv)
            writer.writerow(record)
    except:
        print('#######error########', line[1])

#关闭打开的两个csv
csvTest.close()
resultCsv.close()
            
        



# In[1]:


a = [[1,2,3],[12,34,5]]
print(a.shape)


# In[4]:


import os

for i in os.listdir("C:/Users/91190/Desktop/X-Cov"):
    print(i)


# In[1]:


a="/owjfos/fs/da/fa.png"
s = a.split('/')[-1][0:-4]
print(s)

