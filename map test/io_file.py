# coding=utf-8
import xml.dom.minidom
import json
import csv
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='calculate mAP of IOUs[0.5:0.05:0.95]')
    parser.add_argument('--ground_truth_path',default='underWater/ground_truth_val_test2kimg',help='the path of ground truth')
    parser.add_argument('--detection_path',default='C:/Users/Ruofei/Desktop/metric/underWater/htc_scallop_train_lr0005_epoch_2_val_test2kimg',help='the path of detection')
    parser.add_argument('--annotations_path',default='annotations_val_test2kimg.json',help='the path of annotations of ground truth')
    parser.add_argument('--detect_result_csv_path',default='underwater_data/htc_scallop_train_lr0005_epoch_2_val_test2kimg.csv',help='the path of csv result of detection')
    parser.add_argument('--confidence_th', default='0.0001', help='confidence threshold')
    args = parser.parse_args()
    return args
def parse_csv_to_txt(csv_path,detection_path,confidence_th=0.0001):
    fp_csv=open(csv_path)
    data_csv=csv.reader(fp_csv)
    for line in data_csv:
        bbox_info=''
        if line[0]=='holothurian':
            bbox_info='1'
        elif line[0]=='echinus':
            bbox_info='2'
        elif line[0]=='scallop':
            bbox_info='3'
        elif line[0]=='starfish':
            bbox_info='4'
        elif line[0]=='':
            break
        #print(line)
        if (float(line[2]) < float(confidence_th)):
            continue
        bbox_info=bbox_info+' '+str(line[3])+' '+str(line[4])+' '+str(line[5])+' '+str(line[6])+' '+str(line[2])+'\n'
        img_txt_name=line[1][0:7]+'txt'
        fp_txt = open(detection_path + '/' + img_txt_name, 'a')
        fp_txt.writelines(bbox_info)
        fp_txt.close()
    fp_csv.close()
    print('finished parse csv to txt')

def parse_json_to_txt(json_path,ground_truth_path):
    with open(json_path,'r') as fp_json:
        loaded_json=json.load(fp_json)
        img_name=[]
        for image in loaded_json['images']:
            for key,value in image.items():
                if(key=='file_name'):
                    img_name_temp=value[:7]
                    img_name_temp=img_name_temp+'txt'
                    img_name.append(img_name_temp)
                    fp_ground_truth=open(ground_truth_path+'/'+img_name_temp,'a')
                    fp_ground_truth.close()
        for annotation in loaded_json['annotations']:
            bbox_info = ""
            img_txt_name = ""
            for key,value in annotation.items():
                if(key=='image_id'):
                    image_id=value
                    img_txt_name=img_name[image_id-1]
                elif(key=='bbox'):
                    bbox_info=bbox_info+str(value[0])+' '+str(value[1])+' '+str(value[0]+value[2])+' '+str(value[1]+value[3])
                elif(key=='category_id'):
                    bbox_info=str(value)+' '+bbox_info+'\n'
                    #print(bbox_info)
                    fp_txt = open(os.path.join(ground_truth_path,img_txt_name), 'a')
                    fp_txt.writelines(bbox_info)
                    fp_txt.close()
def make_detection_txt(json_path,detection_path):
    if(os.path.exists(detection_path)!=1):
        print("mkdir ",detection_path)
        os.makedirs(detection_path)
    with open(json_path,'r') as fp_json:
        loaded_json=json.load(fp_json)
        for image in loaded_json['images']:
            for key,value in image.items():
                if(key=='file_name'):
                    img_name_txt=value[:7]
                    img_name_txt=img_name_txt+'txt'
                    fp_detection = open(os.path.join(detection_path,img_name_txt), 'a')
                    fp_detection.close()

def make_ground_truth_txt(json_path,ground_truth_path):
    if(os.path.exists(ground_truth_path)!=1):
        print('mkdir',ground_truth_path)
        os.makedirs(ground_truth_path)
    with open(json_path,'r') as fp_json:
        loaded_json=json.load(fp_json)
        for image in loaded_json['images']:
            for key,value in image.items():
                if(key=='file_name'):
                    img_name_temp=value[:7]
                    img_name_txt=img_name_temp+'txt'
                    fp_ground_truth = open(os.path.join(ground_truth_path,img_name_txt), 'a')
                    fp_ground_truth.close()

# class_map = {0: 'label0', 1: 'label1', 2: 'label2',3:'label3'}
# def parse_xml(xml_path):
#     dom = xml.dom.minidom.parse(xml_path)
#     root = dom.documentElement
#     objects = root.getElementsByTagName('object')
#     gts = []
#     for index, obj in enumerate(objects):
#         name = obj.getElementsByTagName('name')[0].firstChild.data
#         label = class_map[name]
#         bndbox = obj.getElementsByTagName('bndbox')[0]
#         x1 = int(bndbox.getElementsByTagName('xmin')[0].firstChild.data)
#         y1 = int(bndbox.getElementsByTagName('ymin')[0].firstChild.data)
#         x2 = int(bndbox.getElementsByTagName('xmax')[0].firstChild.data)
#         y2 = int(bndbox.getElementsByTagName('ymax')[0].firstChild.data)
#         gt_one = [label, x1, y1, x2, y2]
#         gts.append(gt_one)
#     return gts
def main():
    args=parse_args()
    #make_ground_truth_txt(args.annotations_path, args.ground_truth_path)
    #parse_json_to_txt(args.annotations_path, args.ground_truth_path)
    print('置信度阈值为：{}'.format(args.confidence_th))
    make_detection_txt(args.annotations_path, args.detection_path)
    parse_csv_to_txt(args.detect_result_csv_path, args.detection_path,args.confidence_th)

if __name__ == "__main__":
    main()


