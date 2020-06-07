#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# Author: 杜建军
# @Time: 2020/3/11 21:36
# @IDE: PyCharm
import json
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# 取数据,画扇形统计图，统计每个图片尺寸出现的次数
with open('annotations.json','r') as fp:
    loaded_json=json.load(fp)
    image_size={} #{'image_size':num}
    image_height=0
    image_width=0
    for image in loaded_json['images']:
        image_height=0
        image_width=0
        for key,value in image.items():
            if(key=='height'):
                image_height=value
            if(key=='width'):
                image_width=value
                size_str=str(image_width)+'*'+str(image_height)
                if size_str in image_size:
                    image_size[size_str] = image_size[size_str] + 1
                else:
                    image_size[size_str] = 1
    print(image_size)
#绘制柱状图
x=[]
y=[]
for key_size,value_num in image_size.items():
    x.append(key_size)
    y.append(value_num)
color = ['blue', 'red','green','yellow','pink'];
print(x)
print(y)
label =x

def show_bar():
    plt.bar(np.arange(5)+1,y, width=[1]*5,
            color=color, align='center',
            tick_label=label);
    plt.xlim(0, 6)
    plt.ylim(0, 3500)
    plt.title("image_size_bar")
    plt.savefig("image_size_bar.jpg")
    plt.show()
show_bar()

#绘制饼图
fig = plt.figure()
plt.pie(y, labels=label, autopct='%1.2f%%')  # 画饼图（数据，数据对应的标签，百分数保留两位小数点）
plt.title("image_size_pie")
plt.savefig("image_size_pie.jpg")
plt.show()
