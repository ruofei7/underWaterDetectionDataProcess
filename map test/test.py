# coding=utf-8
from evaluation import *
import sys
import argparse
import numpy as np
import csv


cfg = {'file_dir': [],
       'overlapRatio': 0.5,
       'cls': 4,
       'presicion': False,
       'recall': False,
       'threshold': 0.5,
       'FPPIW': False,
       'roc': False,
       'pr': False}


def parse_args():

    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Detection Results Test!')
    parser.add_argument('-dir', dest='dir', default= ['underWater/ground_test2kimg' ,\
                                                      'underWater/ground_test2kimg'], \
                        nargs='+', help='Two folders with detection results and ground truth in '
                                                            'each of them， put detection path in front', type=str)

    parser.add_argument('-ratio', dest='overlapRatio', help='Should be in [0, 1], float type, which means the IOU '
                                                            'threshold, default = 0.5', default=0.5, type=float)

    parser.add_argument('-thre', dest='threshold', help='Should be in [0, 1], float type, if you need [precision] '
                                                        ', [recall], [FPPI] or [FPPW], default = 0.5', default=0.25, type=float)

    parser.add_argument('-cls', dest='cls', help='Should be > 1, which means number of categories(background included),'
                                                     'default = 2', default=5, type=int)

    parser.add_argument('-prec', dest='precision', help='Should be True or False, which means return precision or not, '
                                                        'default = True', default=False, type=bool)

    parser.add_argument('-rec', dest='recall', help='Should be True or False, which means return recall or not, '
                                                    'default = True', default=False, type=bool)

    parser.add_argument('-FPPIW', dest='FPPIW', help='Should be True or False, which means return FPPI and FPPW or not,'
                                                    'default = True', default=False, type=bool)

    parser.add_argument('-roc', dest='roc', help='Should be True or False, which means drawing ROC curve or not, '
                                                    'default = True', default=False, type=bool)

    parser.add_argument('-pr', dest='pr', help='Should be True or False, which means drawing PR curve or not, '
                                                    'default = True', default=False, type=bool)

    args_in = parser.parse_args()

    return args_in

def test_main():
    args = parse_args()
    # args.dir = ['/Users/wangzhe/data/safe_belt/part1/prediction/', '/Users/wangzhe/data/safe_belt/part1/test_annos/']
    # args.dir = ['underWater/ground_truth_4', 'underWater/val_detection_4']

    args.cls = 4  # 类别数
    # len(sys.argv)
    # print ("Your Folder's path: {}".format(args.dir))
    # print ("Overlap Ratio: {}".format(args.overlapRatio))
    # print ("Threshold: {}".format(args.threshold))
    # print ("Num of Categories: {}".format(args.cls))
    # print ("Precision: {}".format(args.precision))
    # print ("Recall: {}".format(args.recall))
    # print ("FPPIW: {}".format(args.FPPIW))
    #
    print("Calculating......")
    # for model in os.listdir(args.dir[0]+'/'+'val_result'):
    #     model_name=model
    #     model_path=args.dir[0]+'/'+model
    #     for detection in os.listdir(model_path):
    #         #if(detection=='epoch_25_val'):
    #         detection_path=model_path+'/'+detection
    #         epoch_name=detection
    cfg['file_dir'] = [args.dir[0], args.dir[1]]
    print(cfg['file_dir'])
    cfg['overlapRatio'] = args.overlapRatio
    cfg['cls'] = args.cls
    cfg['precision'] = args.precision
    cfg['recall'] = args.recall
    cfg['threshold'] = args.threshold
    cfg['FPPIW'] = args.FPPIW
    cfg['roc'] = args.roc
    cfg['pr'] = args.pr

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

    # all_epoch_csv=open("all_epoch_csv.csv",'a+')
    # writer_csv=csv.writer(all_epoch_csv)
    # writer_csv.writerow([model_name,epoch_name,sum_map/len(IOUs)])
    for key, val in result_eval.items():
        if (key == 1):
            print('class 1 海参平均误检率', sum(result_eval[key]['false_detection']) / len(IOUs),
                  result_eval[key]['false_detection'])
            print('class 1 海参平均漏检率', sum(result_eval[key]['miss_detection']) / len(IOUs),
                  result_eval[key]['miss_detection'])
            print('class 1 海参平均ap', sum(result_eval[key]['ap']) / len(IOUs), result_eval[key]['ap'])
            print('\n')
        elif (key == 2):
            print('\n')
            print('class 2 海胆平均误检率', sum(result_eval[key]['false_detection']) / len(IOUs),
                  result_eval[key]['false_detection'])
            print('class 2 海胆平均漏检率', sum(result_eval[key]['miss_detection']) / len(IOUs),
                  result_eval[key]['miss_detection'])
            print('class 2 海胆平均ap', sum(result_eval[key]['ap']) / len(IOUs), result_eval[key]['ap'])
        elif (key == 3):
            print('\n')
            print('class 3 扇贝平均误检率', sum(result_eval[key]['false_detection']) / len(IOUs),
                  result_eval[key]['false_detection'])
            print('class 3 扇贝平均漏检率', sum(result_eval[key]['miss_detection']) / len(IOUs),
                  result_eval[key]['miss_detection'])
            print('class 3 扇贝平均ap', sum(result_eval[key]['ap']) / len(IOUs), result_eval[key]['ap'])
        elif (key == 4):
            print('\n')
            print('class 4 海星平均误检率', sum(result_eval[key]['false_detection']) / len(IOUs),
                  result_eval[key]['false_detection'])
            print('class 4 海星平均漏检率', sum(result_eval[key]['miss_detection']) / len(IOUs),
                  result_eval[key]['miss_detection'])
            print('class 4 海星平均ap', sum(result_eval[key]['ap']) / len(IOUs), result_eval[key]['ap'])
    print('average mAP:', sum_map / len(IOUs))
    print('IOUs:', IOUs)

    # eval = evaluation(cfg)
    # mAP = eval.run()

if __name__ == "__main__":
   test_main()
