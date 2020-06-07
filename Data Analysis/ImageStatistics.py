#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# Author: 杜建军
# @Time: 2020/3/11 10:37
# @IDE: PyCharm
import json
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# 取数据,画扇形统计图，统计每个类别出现的次数
with open('annotations.json','r') as fp:
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
#绘图
labels = ['0#holothurian', '1#echinus', '2#scallop', '3#starfish']
X = class_num
fig = plt.figure()
plt.pie(X, labels=labels, autopct='%1.2f%%')  # 画饼图（数据，数据对应的标签，百分数保留两位小数点）
plt.title("image_class_num_pie")
plt.savefig("image_class_num_pie.jpg")
plt.show()