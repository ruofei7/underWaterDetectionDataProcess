
# Author: 杜建军
# @Time: 2020/3/21 15:59
# @IDE: PyCharm
# 推理程序，infer test img，并输出csv文件

import csv
import os
from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='test detector & get csv')
    parser.add_argument('--config', default="/home/lyuan/pycharm_code/mmdetection/configs/reppoints/reppoints_moment_x101_dcn_fpn_2x_mt.py",
                        help='train config file path')
    parser.add_argument('--work_dir', default="/home/lyuan/pycharm_code/mmdetection/exp/reppoints_moment_x101_dcn_fpn_2x_mt",
                        help='the dir to save result img')
    parser.add_argument('--epoch_file', default="/home/lyuan/pycharm_code/mmdetection/exp/reppoints_moment_x101_dcn_fpn_2x_mt")
    parser.add_argument('--pathImg', default="/home/lyuan/pycharm_code/mmdetection/data/underwaterdd/val/image",
                        help='the dir to read test img')
    parser.add_argument('--csvPath', default="/home/lyuan/pycharm_code/mmdetection/data/underwaterdd/val_result",
                        help='the dir to save csv')

    args = parser.parse_args()
    return args
def validate_epoch(config_path,work_dir,epoch_path,pathImg,csvPath):
    model = init_detector(config_path, epoch_path, device='cuda:4')

    resultCsv = open(csvPath, "a")  # 打开result csv文件，检测的结果将存放在这里面，需要提前新建好
    writer = csv.writer(resultCsv)
    # 取testImg文件夹中的每一张图片
    #pathSaveImg = csvPath.split('/')[-1][0:-4]
    pathSaveImg=os.path.join(work_dir,'val_result',epoch_path.split('/')[-1][:-4])
    if(os.path.exists(pathSaveImg)!=1):
        os.makedirs(pathSaveImg)
    for imgP in os.listdir(pathImg):
        img =os.path.join(pathImg,imgP) #args.pathImg + '/' + imgP  # img的绝对路径
        #print("#######img\n",img)
        # 将img传给model，产生输出
        # test a single image
        result = inference_detector(model, img)
        # save the visualization results to image files
        csvOut=[]
        #show_result(img, result, model.CLASSES,score_thr=0.0001, show=False, out_file=args.work_dir + '/result_image/' + imgP, out_csv=csvOut)
        show_result(img, result, model.CLASSES, score_thr=0.0001, show=False,\
                    out_file=os.path.join(pathSaveImg,imgP), out_csv=csvOut)

        # 将结果写进csv,result array中保存了很多bbox，一次只写一个
        for aBbox in csvOut:
            # 产成大于阈值的list并将该list命名为record
            writer.writerow(aBbox)
    # 关闭打开的两个csv
    resultCsv.close()
def main():
    args = parse_args()
    for file in os.listdir(args.epoch_path):
        epoch_path=os.path.join(args.epoch_path,file)
        if(epoch_path[-4:]=='.pth'):
            print('epoch_path:',epoch_path)
            csvPath=os.path.join(args.csvPath,args.work_dir.split('/')[-1],epoch_path.split('/')[-1][:-4],'val.csv')
            validate_epoch(args.config_path,args.work_dir,epoch_path,args.pathImg,csvPath)



if __name__ == '__main__':
    main()