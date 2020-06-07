# 对训练集数据进行分析：

xml_to_json.py 将官方给的xml文件转化为json标注文件。



image_size_bar.py 取数据,画扇形统计图，统计训练集中每个图片尺寸出现的次数，用于确定输入图像和测试尺寸。

![image](https://github.com/Ruofei520/underWaterDetectionDataProcess/blob/master/Images/image_size_pie.jpg)



image_test_size_bar.py  统计测试集中每个图片尺寸出现的次数，比较训练集和测试集图像的尺寸分布，发现测试集中有2k图出现，训练集中没有。

![image](https://github.com/Ruofei520/underWaterDetectionDataProcess/blob/master/Images/image_test_B_size_pie.jpg)



image_per_object_num.py 统计训练集中的目标尺寸（检测框的大小）可以更好的调整anchor的大小。

![image](https://github.com/Ruofei520/underWaterDetectionDataProcess/blob/master/Images/bbox_area_bar.jpg)



ImageStatistics.py 统计训练集中每个类别出现的次数，发现类别数量上的差异。

![image](https://github.com/Ruofei520/underWaterDetectionDataProcess/blob/master/Images/image_class_num_pie.jpg)



divide2k.py 将测试集中的2k图和非2k图划分开来

draw_bbox_by_csv.py 通过csv画出对测试集的检测结果，看哪些是画错了的，进行错误分析。根据错误分析，我们发现在2k图上表现很差，所以对2k图进行了标记，对2k图进行单独训练，获得了最佳效果。







