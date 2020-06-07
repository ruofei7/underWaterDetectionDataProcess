#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# Author: 杜建军
# @Time: 2020/3/11 21:36
# @IDE: PyCharm
import json
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import cv2
import argparse
import os

def parse_args():
    parser=argparse.ArgumentParser(description='analise the size of test img')
    parser.add_argument('--img_path',default='E:/UnderWaterDetection/test-B-image/test-B-image')
    parser.add_argument('--save_result_path',default='result')
    args=parser.parse_args()
    return args
def get_size_info():
    args = parse_args()
    if os.path.exists(args.img_path) == False:
        raise AssertionError(args.img_path, ' is not found')
    if os.path.exists(args.save_result_path) == False:
        raise AssertionError(args.save_result_path, ' is not found')
    imgs_size = {}  # {'img_size':num}
    for img_name in os.listdir(args.img_path):
        img_path = os.path.join(args.img_path, img_name)
        img = cv2.imread(img_path)
        img_size = str(img.shape[1]) + '*' + str(img.shape[0])
        print(img_size, img_path)
        if img_size in imgs_size:
            imgs_size[img_size] = imgs_size[img_size] + 1
        else:
            imgs_size[img_size] = 1
    print(imgs_size)
    return imgs_size
def drwa_result(imgs_size={}):
    args = parse_args()
    if os.path.exists(args.save_result_path) == False:
        raise AssertionError(args.save_result_path, ' is not found')
    x = []
    y = []
    for key_size, value_num in imgs_size.items():
        x.append(key_size)
        y.append(value_num)
    #color = ['blue', 'red', 'green', 'yellow', 'pink','black','orange'];
    print(x)
    print(y)
    label = x
    plt.bar(np.arange(7) + 1, y, width=[1] * 7, align='center',
            tick_label=label)
    plt.xlim(0, 7)
    plt.ylim(0, 3500)
    plt.title("image_test_B_size_bar")
    plt.savefig(args.save_result_path + '/' + "image_test_B_size_bar.jpg")
    plt.show()
    # 绘制饼图
    fig = plt.figure()
    plt.pie(y, labels=label, autopct='%1.2f%%')  # 画饼图（数据，数据对应的标签，百分数保留两位小数点）
    plt.title("image_test_B_size_pie")
    plt.savefig(args.save_result_path + '/' + "image_test_B_size_pie.jpg")
    plt.show()
def main():
    #imgs_size=get_size_info()
    #imgs_size={'1080*1920': 42, '1440*2560': 32, '1536*2048': 21, '2160*3840': 653, '405*720': 52}
    imgs_size={'1920*1080': 72, '2560*1440': 42, '2048*1536': 28, '3840*2160': 934, '586*480': 2, '704*576': 3, '720*405': 119}
    drwa_result(imgs_size)



if __name__=='__main__':
    main()
