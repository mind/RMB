# -*- coding: UTF-8 -*-

"""
# version: python3.6
# author: TechDing
# email: dingustb@126.com
# file: extract_ROI.py
# time: 2019/7/2 9:41
# doc: 使用OpenCV图像处理方式来提取RMB图像中编码所在区域（ROI）
"""
import os
MODEL_PATH=os.path.abspath('./CV_ROI/Models/Face_Model.h5')
import cv2
import numpy as np
def model_predict_img(model, img_arr):  # OK
    '''
    使用model来预测img_arr，判断其属于哪一个类别
    :param model: 已经训练好的分类模型
    :param img_arr: 完整图片的ndarray, 0-255,uin8
    :return: 该图片所属类别。
    '''

    def crop_img(img, size=(320, 320)):
        if min(img.shape[:2]) < min(size):  # 如果尺寸不够，则要resize
            img = cv2.resize(img, (max(img.shape[1], size[0]), max(img.shape[0], size[1])))
        H0 = int(img.shape[0] / 2)
        W0 = int(img.shape[1] / 2)  # center point
        half_H = int(size[0] / 2)
        half_W = int(size[1] / 2)
        return cv2.resize(img[H0 - half_H:H0 + half_H, W0 - half_W:W0 + half_W], (int(size[0] / 2), int(size[1] / 2)))

    small_img = np.array([crop_img(img_arr, size=(320, 320))])
    predy = model.predict(small_img.astype('float64') / 255)  # 此处只预测一张图片
    return np.argmax(predy, axis=1)[0]  # 此处只有一张图，所以取第一个即可

from keras.models import load_model
from glob import glob
import os
import keras.backend as K
def get_save_roi(src_folder, save_folder):  # OK
    '''
    首先用model来预测img_path图片所属面值的类别，然后分类获取不同面值图片中的ROI(即编码所在区域),
    并将该ROI图片保存到save_folder中。
    # :param model_path: 已经训练好的面值分类模型的路径
    :param src_folder: 需要获取的图片所在文件夹
    :param save_folder: 获取的ROI保存的文件夹，文件名为该图片的文件名
    :return: None
    '''

    def func1_01(img):  # 0.1元
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def func2_01(img):  # 0.1元
        return img[-80:, 260:500]

    def func1_02(img):  # 0.2元
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=1)

    def func2_02(img):
        return img[-80:, 270:510]

    def func1_05(img):
        img1 = cv2.GaussianBlur(img, (5, 5), 0)
        return cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    def func2_05(img):
        return img[-80:, 150:390]

    def func1_1(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=3)
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=3)
        gray = cv2.equalizeHist(gray)
        return gray

    def func2_1(img):
        img = img[-100:, 450:690]
        return cv2.resize(img, (240, 80))

    def func1_2(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=4)
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=4)
        return cv2.equalizeHist(gray)

    def func2_2(img):
        img = img[-100:, 430:690]
        return cv2.resize(img, (240, 80))

    def func1_5(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=2)

    def func2_5(img):
        return img[-95:-15, 460:700]

    def func1_10(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=5)
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=5)
        return cv2.equalizeHist(gray)

    def func2_10(img):
        img = img[-100:, 460:700]
        return cv2.resize(img, (240, 80))

    def func1_50(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=3)
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=3)
        return cv2.equalizeHist(gray)

    def func2_50(img):
        return img[200:280, -240:]

    def func1_100(img):  # 与50元的处理方式一样
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=3)
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=3)
        return cv2.equalizeHist(gray)

    def func2_100(img):
        return img[200:280, -240:]

    def func2_default(img):
        roi = img[-120:-40, 580:870]
        return cv2.resize(roi, (240, 80))

    def calc_roi(img_arr, func1, func2):
        '''
        计算某图片的roi，该roi图片为（80,240,3）
        :param img_arr: BGR图片，cv2.imread得到
        :param func1: 图片处理的辅助函数
        :param func2：截取图片的辅助函数
        :return: roi
        '''

        def getAffine(img_arr, src_points):
            dst_points = np.float32([[0, 400], [0, 0], [872, 0]])
            affineMatrix = cv2.getAffineTransform(src_points, dst_points)
            return cv2.warpAffine(img_arr, affineMatrix, (872, 400))

        gray = func1(img_arr)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        _, contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        found = False
        for contour in contours:
            rect = cv2.minAreaRect(contour)  # 获取最小外接矩形
            width, height = rect[1]  # 长宽
            short = min(width, height)
            long = max(width, height)
            if short < min(img_arr.shape[:2]) * 0.5 or long < max(img_arr.shape[:2]) * 0.5 or short >= min(
                    img_arr.shape[:2]) or long >= max(img_arr.shape[:2]):
                continue
            box = cv2.boxPoints(rect)  # 获取最小外接矩形的4个顶点坐标
            box = np.int0(box)
            # cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
            found = True
            src_points = np.float32(box[1:]) if rect[2] < -45 else np.float32(box[:3])
            changed = getAffine(img_arr, src_points)
            break  # 如果找到了一个就不用在继续寻找
        return func2(changed) if found else func2_default(img_arr)

    model_path=MODEL_PATH
    all_imgs = glob(src_folder + '/*.jpg')  # 原始图片都是jpg
    model = load_model(model_path)  # 模型加载非常耗时，需要注意
    dict1 = dict(
        zip(np.arange(9), [func1_01, func1_02, func1_05, func1_1, func1_2, func1_5, func1_10, func1_50, func1_100]))
    dict2 = dict(
        zip(np.arange(9), [func2_01, func2_02, func2_05, func2_1, func2_2, func2_5, func2_10, func2_50, func2_100]))
    for idx, img_path in enumerate(all_imgs):
        img0 = cv2.imread(img_path)
        pred = model_predict_img(model, img0)  # 模型预测出来的类别，0-8
        roi = calc_roi(img0, dict1[pred], dict2[pred])
        _, img_name = os.path.split(img_path)
        cv2.imwrite(os.path.join(save_folder, img_name), roi)
        print('\r{}/{} finished...'.format(idx + 1, len(all_imgs)), end=' ')
    print('all cv_rois are saved to {}'.format(save_folder))
    K.clear_session()
