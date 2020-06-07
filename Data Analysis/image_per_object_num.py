#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# Author: 杜建军
# @Time: 2020/3/11 23:03
# @IDE: PyCharm
import json
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# 取数据,画扇形统计图，统计每个类别出现的次数
with open('annotations.json','r') as fp:
    loaded_json=json.load(fp)
    image_per_object_num={} #{'image_per_object_num':num}
    bbox_area={} #{'box_area':num}
    for annotation in loaded_json['annotations']:
        image_id=-1
        image_change_flag=0
        for key,value in annotation.items():
            if(key=='image_id'):
                image_id=value
            elif(key=='area'):
                bbox_area_insert_flag=0
                for key_area,value_num in bbox_area.items():
                    if(key_area==value):
                        bbox_area[key_area]=value_num+1
                        bbox_area_insert_flag=1
                        break
                if(bbox_area_insert_flag==0):
                    bbox_area[value]=1
                    bbox_area_insert_flag=1

print(bbox_area)
sort_key=sorted(zip(bbox_area.keys(),bbox_area.values()))
sort_value=sorted(zip(bbox_area.values(),bbox_area.keys()),reverse=True)
fw=open("sort_bbox_area.txt",'w+')
fw.write(str(sort_key))
fw.write(str(sort_value))
print(sort_key)
print(sort_value)
#for element in sort_key:
count=5000 #因为尺寸太多了，所以只选同尺寸最多的count种尺寸
draw_count_point={}
for element in sort_value:
    draw_count_point[element[1]]=element[0]
    count-=1
    if(count==0):
        break
print(draw_count_point)
sort_count_key=sorted(zip(draw_count_point.keys(),draw_count_point.values()))
print(sort_count_key)
list_count_key=[]
list_count_value=[]
for element in sort_count_key:
    list_count_key.append(element[0])
    list_count_value.append(element[1])

#绘制柱状图
x=list_count_key
y=list_count_value

def show_bar():
    plt.bar(np.arange(len(x))+700,y, width=[0.6]*len(x), align='center',
            tick_label=x)
    plt.xlim(0, len(x)*2)
    plt.ylim(0, 80)
    plt.title("bbox_area_bar")
    plt.savefig("bbox_area_bar.jpg")
    plt.show()

show_bar()

