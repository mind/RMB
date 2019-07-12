# -*- coding: UTF-8 -*-

"""
# version: python3.6
# author: TechDing
# email: dingustb@126.com
# file: Utils.py
# time: 2019/7/2 10:22
# doc: CTC_Models模块中的各种辅助函数
"""

import numpy as np
from glob import glob
import cv2,os
def load_roi_setX(roi_folder,save_X_path,save_imgs_path,size=(80,240)):
    '''
    加载所有的roi图片为roiX.npy并保存到save_X_path
    :param roi_folder: 所有的roi都保存到这个文件夹中
    :param save_X_path: roiX.npy保存的路径
    :param save_imgs_path: 图片名称对应的.npy保存路径
    :parma size: roi会经过整理为统一的一个尺寸，此处为（80,240）
    :return:
    '''
    roi_H, roi_W=size
    def get_img_arr(img_path):
        '''
        读取图片img_path,读取后的shape为（roi_H,roi_W,3）
        :param img_path:
        :return:
        '''
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        if h != roi_H or w != roi_W:
            img = cv2.resize(img, (roi_W, roi_H))
        return img
    all_imgs=glob(roi_folder+'/*.jpg')
    all_data = np.zeros(shape=(len(all_imgs), roi_H, roi_W, 3), dtype='uint8')  # 存储图片 (bs, 80,240,3)
    img_names=[]
    for idx, img_path in enumerate(all_imgs):
        all_data[idx] = get_img_arr(img_path)
        _, img_name = os.path.split(img_path)
        img_names.append(img_name)
        print('\r{}/{} finished...'.format(idx + 1, len(all_imgs)), end=' ')
    np.save(save_X_path, all_data)
    np.save(save_imgs_path,np.array(img_names))


from collections import OrderedDict
import re
from collections import Counter
def ensemble(cv_result_path, ctpn_result1_path, ctpn_result2_path, save_path):
    '''
    对三个csv结果进行集成处理，三个结果中取个数最多的为最终结果。
    并将最终结果保存到save_path中
    :param cv_result_path:
    :param ctpn_result_path:
    :param ctpn_result2_path:
    :param save_path:
    :return:
    '''
    def load_result(txt_path):
        result_dict = OrderedDict()
        with open(txt_path, 'r') as file:
            lines = file.readlines()
        for line in lines:
            name, code = line.strip().split(',')
            result_dict[name] = code.strip()
        return result_dict

    def valid_str(strA, rep_str='?'):
        def get_num(str_, ch='num'):
            if ch == 'num':
                return len(re.compile('[0-9]').findall(str_))
            else:
                return len(re.compile('[A-Z]').findall(str_))

        last_num = get_num(strA[3:], 'num')
        header_num = get_num(strA[:3], 'str')
        if len(strA) == 10 and last_num == 7 and header_num == 2:
            return strA  # strA 格式正确，直接返回

        if last_num < 7:
            last_str = re.sub(re.compile('[A-Z]'), rep_str, strA[3:])
            strA=strA[:3] + last_str if header_num==2 else strA[:3] + last_str.rstrip(rep_str)
        num_str = [str(i) for i in range(10)]
        if strA[0] in num_str:
            strA = rep_str + strA[1:]  # 如果以数字开头，数字替换为?
        str_len = len(strA)
        if str_len < 10: # 如果总长度不足10个，前面用？填充
            header = rep_str * (10 - str_len)
            strA = header + strA
        return strA

    def get_compare(str1, str2, str3):
        def is_char_same(strA, strB, strC):
            return [(strA[i] == strB[i] and strB[i] == strC[i]) for i in range(len(strA))]

        str1 = valid_str(str1)
        str2 = valid_str(str2)
        str3 = valid_str(str3)
        is_same = is_char_same(str1, str2, str3)
        combined_str = []
        for idx, same in enumerate(is_same):
            if same:  # 如果三个字符都相同
                temp_ch = 'A' if idx < 3 else '2'
                temp_ch= temp_ch if str1[idx] == '?' else str1[idx]
                combined_str.append(temp_ch)
            else:  # 如果三个字符都不同，获取每个字符的个数
                count = Counter([str1[idx], str2[idx], str3[idx]])
                temp = sorted(count.items(), key=lambda x: x[1], reverse=True)
                ch = temp[1][0] if temp[0][0] == '?' else temp[0][0]
                if temp[0][1] == 1:  # 如果个数都是1个，先考虑str3,在考虑str2,最后设为0
                    ch=str3[idx] if str3[idx] !='?' else str2[idx] if str2[idx] !='?' else '0'
                combined_str.append(ch)
        return ''.join(combined_str)

    ctc_dict = load_result(cv_result_path)
    ctpn_dict1 = load_result(ctpn_result1_path)
    ctpn_dict2 = load_result(ctpn_result2_path)
    new_dict = OrderedDict()
    new_dict['name'] = 'label'
    default_value = '?' * 10
    for key, value in ctc_dict.items():
        if key == 'name': continue
        ctc_value = value  # 初始化
        ctpn_value1 = ctpn_dict1.get(key, default_value) # ctpn_dict1的长度可能偏少，有些图片没有找到ROI
        ctpn_value2 = ctpn_dict2.get(key, default_value)
        if ctpn_value1.endswith('Z'):  # 如果结果最后一位为Z，代表结果预测不准，全部用?代替。
            ctpn_value1 = default_value
        if ctpn_value2.endswith('Z'):
            ctpn_value2 = default_value
        real_value = get_compare(ctc_value, ctpn_value1, ctpn_value2)
        new_dict[key] = real_value

    def write_dict_result(dict_result, save_path):
        if 'name' in dict_result.keys():
            dict_result.pop('name')
        with open(save_path, 'w') as file:
            file.write('name,label\n')
            for key, value in dict_result.items():
                file.write(key + ',' + value + '\n')

    write_dict_result(new_dict, save_path) # 将最终结果保存到save_path中
    print('finished. Ensembled result is saved to {}'.format(save_path))

