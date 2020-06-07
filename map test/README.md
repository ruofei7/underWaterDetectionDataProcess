### 由于mmdetection中测map一直有问题，测出来都是0，因此进行线下测试。
 inference.py 替换mmdetection中的inference.py，在inference.py中加入了test_out函数，用于拿到检测结果。
 io_file.py 用于将ground truth的annotation生成txt，将检测结果的csv生成txt,因为test.py是对txt文件进行map计算。
 测试的时候，先运行io_file.py生成txt文件，再用test.py进行计算。
 
 大部分情况下，会对很多个epoch的检测结果进行评测，每个都这个弄比较麻烦，因此可以用validate_per_epoch.py对多个结果进行评测。

检测结果图：
![image](https://github.com/Ruofei520/underWaterDetectionDataProcess/blob/master/Images/mapTestResult.png)
