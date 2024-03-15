import PIL

from data.load_data import CHARS, CHARS_DICT, LPRDataLoader, cv_imread
from PIL import Image, ImageDraw, ImageFont
from model.LPRNet import build_lprnet
# import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import *
from torch import optim
import torch.nn as nn
import numpy as np
import argparse
import torch
import time
import cv2
import os
from torchvision import transforms
import gradio as gr


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--img_size', default=[94, 24], help='the image size')
    parser.add_argument('--test_img', default="./data/output/output.jpg", help='the test images')
    parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
    parser.add_argument('--pretrained_model', default='./weights/Final_LPRNet_model.pth', help='pretrained base model')
    args = parser.parse_args()
    return args


args = get_parser()
lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=0, class_num=len(CHARS))
device = torch.device("cuda:0" if args.cuda else "cpu")
lprnet.to(device)
print("Successful to build network!")
# load pretrained model
if args.pretrained_model:
    lprnet.load_state_dict(torch.load(args.pretrained_model))
    print("load pretrained model successful!")
else:
    print("[Error] Can't found pretrained mode, please check!")
    exec()

def transform(img):
    img = img.astype('float32')
    img -= 127.5
    img *= 0.0078125
    img = np.transpose(img, (2, 0, 1))
    return img

def detect_plate(img):

    # convert input image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # read haarcascade for number plate detection
    cascade = cv2.CascadeClassifier('./model/haarcascade_russian_plate_number.xml')

    # Detect license number plates
    plates = cascade.detectMultiScale(gray, 1.2, 5)
    # print('Number of detected license plates:', len(plates))

    color_plates_list = []
    # loop over all plates
    for (x, y, w, h) in plates:
        # draw bounding rectangle around the license number plate
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        gray_plates = gray[y:y + h, x: + w]
        color_plates = img[y:y + h, x:x + w]
        color_plates_list.append(color_plates)
    cv2.imwrite('data/output/output.jpg', cv2.cvtColor(color_plates_list[0], cv2.COLOR_BGR2RGB))
    return color_plates_list[0]

def cv2ImgAddText(img, text, pos, textColor=(255, 0, 0), textSize=12):
    if (isinstance(img, np.ndarray)):  # detect opencv format or not
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("data/NotoSansCJK-Regular.ttc", textSize, encoding="utf-8")
    draw.text(pos, text, textColor, font=fontText)

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def show_res(img, label):
    img = np.transpose(img, (1, 2, 0))
    img *= 128.
    img += 127.5
    img = img.astype(np.uint8)

    lb = ""
    for i in label:
        lb += CHARS[i]
    # img = cv2.putText(img, lb, (0,16), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0, 0, 255), 1)
    img = cv2ImgAddText(img, lb, (0, 0))
    return img, lb
    # cv2.imshow("test", img)
    # print("predict: ", lb)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

def infer(Net, img: torch.tensor):
    origin_img = img.numpy().copy()
    if args.cuda:
        img = Variable(img.cuda())
    else:
        img = Variable(img)
    # forward
    prebs = Net(img)
    # greedy decode
    prebs = prebs.cpu().detach().numpy()
    preb_labels = list()
    for i in range(prebs.shape[0]):
        preb = prebs[i, :, :]
        preb_label = list()
        for j in range(preb.shape[1]):
            preb_label.append(np.argmax(preb[:, j], axis=0))
        no_repeat_blank_label = list()
        pre_c = preb_label[0]
        if pre_c != len(CHARS) - 1:
            no_repeat_blank_label.append(pre_c)
        for c in preb_label:  # dropout repeate label and blank label
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        preb_labels.append(no_repeat_blank_label)
    return origin_img[0], preb_labels[0]


def Process(img):
    tImage = detect_plate(img)
    tImage = cv2.cvtColor(tImage, cv2.COLOR_BGR2RGB)
    # 将BGR图像转换为HSV色彩空间
    # hsv = cv2.cvtColor(tImage, cv2.COLOR_BGR2HSV)
    # # 定义黄色的HSV范围
    # # 注意：HSV范围可能需要根据实际图像进行调整
    # lower_yellow = np.array([20, 100, 100])  # 黄色下限
    # upper_yellow = np.array([30, 255, 255])  # 黄色上限
    # yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # # 创建蓝色图像，与原始图像尺寸相同
    # blue_image = np.zeros_like(tImage)
    # # 设置蓝色通道的值为255，其他通道为0，得到纯蓝色图像
    # blue_image[:, :, 2] = 255
    # # 将原始图像中黄色区域替换为蓝色
    # tImage[yellow_mask > 0] = blue_image[yellow_mask > 0]
    # cv2.imwrite('data/output/output.jpg', tImage)
    # cv2.imwrite('data/output/output.jpg', Image)
    # rgb_image = cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)
    # pil_image = PIL.Image.fromarray(rgb_image)
    # pil_image.save('data/output/output.jpg')
    # tImage = cv2.imread('data/output/output.jpg')
    res0 = tImage
    height, width, _ = tImage.shape
    if height != args.img_size[1] or width != args.img_size[0]:
        Image = cv2.resize(tImage, args.img_size)
    Image = transform(Image)
    tensor_img = torch.from_numpy(Image)
    new_shape = (1,) + tensor_img.shape
    # 使用view方法改变张量的形状
    tensor_img = tensor_img.view(new_shape)
    origin_img, inferlabel = infer(lprnet, tensor_img)
    img, label = show_res(origin_img, inferlabel)
    return res0, img, label


gr.Interface(
    fn=Process,
    #按照处理程序设置输入组件
    inputs=["image"],
    #按照处理程序设置输出组件
    outputs=["image", "image", "text"],
    examples=[['./plates/car.png'], ['./plates/car2.jpg'], ['./plates/car3.jpg'], ['./plates/car4.jpg'], ['./plates/car5.jpg'], ['./plates/car6.jpg'], ['./plates/car7.jpg']],
).launch(server_name="0.0.0.0")
