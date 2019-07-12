#!/usr/bin/python
# -*- coding: UTF-8 -*-
import cv2
import os

#预测结果的txt文件
f = open("./result_rcnn.txt",'r')
#待预测的图片
ori_root = '../private_test_data'
#切片后图片存储的地址
save_root = '../juesai_t'
for line in f.readlines():
    line_split = line.strip().split(' ')
    pic_name = line_split[0]
    pic_path = os.path.join(ori_root,pic_name)
    pic_cor = line_split[1].split(',')
    img = cv2.imread(pic_path)
    img_split = img[int(pic_cor[1]):int(pic_cor[3]),int(pic_cor[0]):int(pic_cor[2])]
    save_path = os.path.join(save_root,pic_name)
    cv2.imwrite(save_path,img_split)
    print('{} is saved!'.format(pic_name))
