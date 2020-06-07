# 构建list
# 从txt中读取
# 调用WBF，重新创建txt，并将融合后的结果写入

import cv2
import os
from ensemble_boxes import *
from evaluation_wbf import *
import sys
import argparse
import numpy as np
import csv
import shutil
import datetime
import winsound
def wbf_set_weight():
    weights = [1, 1, 2, 2, 3]  # 各个模型的权重
    iou_thr = 0.8  # iou阈值，当iou大于此值时，将两个框归为同一类
    skip_box_thr =0.0001  # 置信度阈值，当置信度低于此值时无视该框default=0.0001
    mktxt = True
    # for i in skip_box_thr:
    #     wbf(weights,iou_thr,skip_box_thr=i,mktxt=mktxt)
    wbf(weights, iou_thr, skip_box_thr=skip_box_thr, mktxt=mktxt)
def wbf(weights,iou_thr=0.8,skip_box_thr=0.0001,mktxt=True):
    now_time = datetime.datetime.now()
    print('line18',now_time)
    path = "big_net_best"  # txt的一级目录，存放待融合的大网络的txt
    pathVal = "C:/Users/Ruofei/Desktop/underWater_others/under_water_data/val_test2kimg/image"  # 验证集图片的目录
    newPath = "wbf_result/wbf_weight"  # 存放融合后的txt的文件夹
    conf_type='max'
    net_type='wbf'
    all_weight_result_path='wbf_result/all_weight_result.csv'

    modelName = []
    for model in os.listdir(path):
        modelName.append(model)
    print(modelName)

    str_weight = ''
    for i in weights:
        str_weight += '_' + str(i)
    print(weights)
    # print(str_weight)
    newPath += str_weight
    newPath=newPath+'_iou_'+str(iou_thr)+'_skip'+str(skip_box_thr)+'_'+conf_type+'_'+net_type
    if (mktxt == True):
        if (os.path.exists(newPath) != True):
            os.mkdir(newPath)
            print('mkdir ', newPath)
        for txtFile in os.listdir(path + '/' + modelName[0]):
            img = cv2.imread(pathVal + '/' + txtFile.replace('txt', 'jpg'))
            [H, W, C] = img.shape
            boxImg = []
            scoreImg = []
            labelImg = []
            for i in range(len(modelName)):
                boxModel = []
                scoreModel = []
                labelModel = []
                with open(path + '/' + modelName[i] + '/' + txtFile, 'r') as f:
                    data = f.readlines()
                    for line in data:
                        curLine = list(map(float, line.split()))
                        curLine[1] /= W
                        curLine[3] /= W
                        curLine[2] /= H
                        curLine[4] /= H
                        boxModel.append(curLine[1:5])  # label xmin ymin xmax ymax confidence；x-长，y-高
                        scoreModel.append(curLine[-1])
                        labelModel.append(curLine[0])
                boxImg.append(boxModel)
                scoreImg.append(scoreModel)
                labelImg.append(labelModel)
            boxes, scores, labels = weighted_boxes_fusion(boxImg, scoreImg, labelImg,
                                                          weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr,conf_type='max')
            fNew = open(newPath + '/' + txtFile, 'a')
            for j in range(len(labels)):
                newLine = str(int(labels[j])) + ' ' + str(int(boxes[j][0] * W)) + ' ' + str(
                    int(boxes[j][1] * H)) + ' ' + str(int(boxes[j][2] * W)) + ' ' + str(int(boxes[j][3] * H)) + ' ' + str(
                    scores[j])
                fNew.write(newLine)
                fNew.write('\n')
        now_time = datetime.datetime.now()
    print('line75', now_time)
    wbf_test(newPath,all_weight_result_path)
    now_time = datetime.datetime.now()
    print('line78', now_time)
    #程序执行结束，发出警报
    duration = 3000  # millisecond
    freq = 440  # Hz
    winsound.Beep(freq, duration)

def wbf_test(detection_path,all_epoch_path,ground_truth_path='underWater/ground_truth_val_test2kimg'):# class_num类别数+1
        cfg = {'file_dir': [],
               'overlapRatio': 0.5,
               'cls': 5,  # 类别数+1
               'precision': False,
               'recall': False,
               'threshold': 0.1,
               'FPPIW': False,
               'roc': False,
               'pr': False}
        print("Calculating......")
        test_info = []
        test_info.append(detection_path.split('/')[-1])
        cfg['file_dir'] = [detection_path, ground_truth_path]
        print(cfg['file_dir'])
        IOUs = np.arange(0.5, 1, 0.05)
        mAPs = []
        sum_map = 0
        result_eval = {1: {'false_detection': [],
                           'miss_detection': [],
                           'ap': []},
                       2: {
                           'false_detection': [],
                           'miss_detection': [],
                           'ap': []},
                       3: {
                           'false_detection': [],
                           'miss_detection': [],
                           'ap': []},
                       4: {
                           'false_detection': [],
                           'miss_detection': [],
                           'ap': []},
                       }
        for IOU in IOUs:
            cfg['overlapRatio'] = IOU
            eval = evaluation(cfg)
            mAP = eval.run(result_eval)
            sum_map += mAP
        test_info.append('mAP')
        test_info.append(sum_map / len(IOUs))
        for key, val in result_eval.items():
            false_detection = sum(result_eval[key]['false_detection']) / len(IOUs)
            miss_detection = sum(result_eval[key]['miss_detection']) / len(IOUs)
            ap = sum(result_eval[key]['ap']) / len(IOUs)
            if (key == 1):
                class_name = 'holothurian'
            elif (key == 2):
                class_name = 'echinus'
            elif (key == 3):
                class_name = 'scallop'
            elif (key == 4):
                class_name = 'starfish'
            else:
                raise AssertionError('out of class names')
            print('class ' + str(key) + ' ' + class_name + ' 平均误检率', false_detection,
                  result_eval[key]['false_detection'])
            test_info.append(class_name + ' false_detection')
            test_info.append(false_detection)
            print('class ' + str(key) + ' ' + class_name + ' 平均漏检率', miss_detection, result_eval[key]['miss_detection'])
            test_info.append(class_name + ' miss_detection')
            test_info.append(miss_detection)
            print('class ' + str(key) + ' ' + class_name + '平均ap', result_eval[key]['ap'])
            test_info.append(class_name + ' ap')
            test_info.append(ap)
            print('\n')
        print('average mAP:', sum_map / len(IOUs))
        print('IOUs:', IOUs)
        all_epoch_csv = open(all_epoch_path, 'a+')
        writer_csv = csv.writer(all_epoch_csv)
        writer_csv.writerow(test_info)
        all_epoch_csv.close()
        print(test_info)

if __name__=='__main__':
    wbf_set_weight()
