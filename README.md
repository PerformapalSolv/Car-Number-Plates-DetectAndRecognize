#  Car-Number-Plates-DetectAndRecognize

思路：**使用python-opencv:cv2.CascadeClassifier** 做车牌检测  、**用LPRNet项目做车牌OCR文字识别**

**车牌识别**

```shell
python number_plate.py
```

**车牌OCR**

```SHELL
python OCR.py
```

CascadeClassifier模型：haarcascade_russian_plate_number.xml  (路径：Car-Number-Plates-DetectAndRecognize/model/haarcascade_russian_plate_number.xml。这是一个基于俄罗斯车牌数据集训练的Haar级联分类器)

[LPRNet项目源码](https://github.com/sirius-ai/LPRNet_Pytorch)

## 项目结果

```shell
python GradioUI.py
```



![image-20240315233532960](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20240315233532960.png)

## 第一版评价

这是我花半天时间完成的项目：这些方法以前了解过，所以找起来、用起来比较快。但由于第二天就要演示了，我没有自己找数据训练模型。

**如：**

1. haarcascade_russian_plate_number.xml模型终究按俄罗斯汽车车牌训练的，很多中国车牌不能检测出。我在gradio上的示例都是挑好的展示，其实大部分是检测不出的。
2. LPRNet也是使用原作者的训练模型：而这个模型的车牌训练集各省份车牌数量占比不平衡，会导致模型对某些省份车牌更敏感，另一些少的会OCR出错。