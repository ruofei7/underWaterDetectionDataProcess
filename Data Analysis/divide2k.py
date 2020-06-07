import cv2
import os

path = 'E:/UnderWaterDetection/test-B-image/test-B-image'#测试机图片路径
path2k = 'E:/UnderWaterDetection/test_B_2k'#新建一个放2k图的文件夹
pathNo2k = 'E:/UnderWaterDetection/test_B_no_2k'#新建一个文件夹放非2k图

for imgName in os.listdir(path):
    img = cv2.imread(path+'/'+imgName)
    # print(img.shape)
    h = img.shape[0]
    w = img.shape[1]
    if (w==2560 and h==1440) or (w==2048 and h==1536):
        cv2.imwrite(path2k+'/'+imgName,img)
    else:
        cv2.imwrite(pathNo2k+'/'+imgName, img)
