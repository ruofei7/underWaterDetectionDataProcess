import cv2
import numpy as np
import argparse
import json
import os
import winsound

def parse_args():
    parser=argparse.ArgumentParser(description='draw bbox by csv')
    parser.add_argument('--imgs_path',default='E:/UnderWaterDetection/test-B-image/test-B-image',help='imgs want to visible bbox')
    parser.add_argument('--save_path',default='draw_bbox')
    parser.add_argument('--csv_path',default='draw_bbox/4703_htc_train_ep12_train_all_ep2-wh101_test_B.csv')
    parser.add_argument('--confidence_th',default=0.3)
    args=parser.parse_args()
    return args

def main():
    args = parse_args()
    if os.path.exists(args.imgs_path)==False:
        print('mkdir ',args.imgs_path)
    if os.path.exists(args.save_path)==False:
        print('mkdir',args.save_path)
    if os.path.exists(args.csv_path)==False:
        raise AssertionError(args.csv_path,'is not found')
    # json_path=args.annotations_path
    # imgs_path=args.imgs_path
    # save_path=args.save_path

    with open(args.csv_path, 'r') as fp_csv:
        for row in fp_csv:
            row_info = row.strip('\n').split(',')[:]
            if float(row_info[2])<args.confidence_th:
                continue
            img_name = row_info[1][:-4] + '.jpg'
            img_path=args.imgs_path+'/'+img_name
            img=cv2.imread(img_path)
            w = img.shape[1]
            save_path_parent=args.save_path+'/'+args.csv_path.split('/')[-1][:-4]+'_'+str(args.confidence_th)+'/'+str(w)
            save_path=save_path_parent+'/'+img_name
            if os .path.exists(save_path):
                img=cv2.imread(save_path)
            img_draw=cv2.rectangle(img,(int(row_info[3]),int(row_info[4])),(int(row_info[5]),int(row_info[6])),(0,0,255),2)
            img_labeled=cv2.putText(img_draw,row_info[0],(int(row_info[3]),int(row_info[4])-3),cv2.FONT_HERSHEY_COMPLEX,fontScale=0.7,color=(0,0,255))
            if os.path.exists(save_path_parent)!=True:
                os.makedirs(save_path_parent)
                print('makedirs',save_path_parent)
            cv2.imwrite(save_path,img_labeled)
            #rint(save_path)
if __name__=='__main__':
    main()
    # 程序执行结束，发出警报
    duration = 1500  # millisecond
    freq = 440  # Hz
    winsound.Beep(freq, duration)