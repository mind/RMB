
"""
# version: python2.7-python3.6
# author: DingRui
# file: rmb_db.py
# time: 2019/6/21 10:10
# doc: 
"""

import os
import sys
import numpy as np
import scipy.sparse
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir) # E:\PyProjects\Codes\CHINESE-OCR\ctpn
try:
    import cPickle as pickle
except:
    import pickle
import uuid
import scipy.io as sio
import xml.etree.ElementTree as ET
from lib.datasets.imdb import imdb

from lib.datasets.ds_utils import *
from lib.fast_rcnn.config import cfg

class rmb_db(imdb):
    def __init__(self,image_set):
        '''
        自定义的数据集，dataset_path必须是一个目录，里面必须有两个文件夹：
        'Images',里面放置最原始的图片，不能含有其他无关文件或损坏的图片文件
        'Labels'，里面放置xml文件，注，首先用LabelImg工具标注，然后将标注框分割成宽16px的切分框，
        此xml里面保存的是这些小切分框的坐标
        :param dataset_path: 该目录的绝对路径
        '''
        imdb.__init__(self, self.__class__.__name__) # 以本类名作为数据集名称
        self._data_path=cfg.DATA_DIR
        self._image_set = image_set
        assert os.path.exists(self._data_path), 'Path does not exist: {}'.format(self._data_path)
        self._classes = ( # 本数据集包含的类别：bg和文本，一共两个类别
            '__background__',
            'text')

        self._class_to_ind = dict( list(zip(self._classes, list(range(self.num_classes)))))
        # self._class_to_ind = {'__background__':0,'text':1}
        self._image_ext='.jpg' # Images文件夹下所有图片都是这个后缀
        self._image_index = self._load_image_set_index()
        # 获取所有图片的名称组成的list，不带.jpg等后缀
        self._roidb_handler = self.gt_roidb # 获取所有图片的xml文件中的信息，一个list，每个元素都是dict，包含xml中的信息
        self._salt = str(uuid.uuid4()) # 产生随机数：eg: 'f7dfd14b-f6c5-4f23-a51a-65bd49143588'
        self._comp_id = 'comp4' # compare_id

        # PASCAL specific config options
        self.config = {
            'cleanup': True,
            'use_salt': True, # compare id构建时是否要加入uuid产生的随机数
            'use_diff': False,
            'matlab_eval': False,
            'rpn_file': None,
            'min_size': 2
        }

    def image_path_at(self, i):
        '''
        获取所有图片列表中第i位置处的图片的绝对路径
        :param i: int， 0-based
        :return: 第i处图片的绝对路径
        '''
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index): # OK
        '''
        获取名称为index的图片路径，该图片的后缀为self._image_ext.
        :param index: 图片的名称，不带后缀
        :return: 返回该图片的绝对路径
        '''
        image_path=os.path.join(self._data_path,'JPEGImages',index + self._image_ext)
        # 获取名称为index的图片的完整路径
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self): # OK
        '''
        从_data_path中读取所有的图片名称（不带后缀）
        :return:返回所有图片名称组成的list
        '''
        # return [os.path.splitext(name_ext)[0] for name_ext in os.listdir(self._data_path+'/Images')]
        image_set_file = self._data_path + '/ImageSets/Main/' + self._image_set + '.txt'
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def gt_roidb(self):
        '''
        首先判断cache文件夹下有没有self.name + '_gt_roidb.pkl'这个文件，如果有则加载返回，
        如果没有，则从所有'Labels'文件夹中加载gt_bbox信息，然后保存到cache文件夹下，返回。
        cache文件夹路径为：根目录(ctpn)下'data/cache'
        :return: 返回获取的roidb文件
        '''
        # name是本数据集的名称
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        # 如果cache存在则直接读取cache
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        # 如果cache不存在，则需要建立cache
        gt_roidb = [
            # 获取所有图片对应的gt_xml中的内容，所有图片的gt_xml组成的list，
            # 每个元素是一个dict，来源于 self._load_pascal_annotation
            self._load_pascal_annotation(index) for index in self.image_index
        ]
        with open(cache_file, 'wb') as fid: # 将获取的图片xml信息保存到cache中
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt_roidb to cache file: {}'.format(cache_file))
        return gt_roidb

    # def selective_search_roidb(self):
    #     """
    #     Return the database of selective search regions of interest.
    #     Ground-truth ROIs are also included.
    #
    #     This function loads/saves from/to a cache file to speed up future calls.
    #     """
    #     cache_file = os.path.join(self.cache_path,
    #                               self.name + '_selective_search_roidb.pkl')
    #
    #     if os.path.exists(cache_file):
    #         with open(cache_file, 'rb') as fid:
    #             roidb = pickle.load(fid)
    #         print('{} ss roidb loaded from {}'.format(self.name, cache_file))
    #         return roidb
    #
    #     if int(self._year) == 2007 or self._image_set != 'test':
    #         gt_roidb = self.gt_roidb()
    #         ss_roidb = self._load_selective_search_roidb(gt_roidb)
    #         roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
    #     else:
    #         roidb = self._load_selective_search_roidb(None)
    #     with open(cache_file, 'wb') as fid:
    #         pickle.dump(roidb, fid, pickle.HIGHEST_PROTOCOL)
    #     print('wrote ss roidb to {}'.format(cache_file))
    #
    #     return roidb
    #
    # def rpn_roidb(self):
    #     if int(self._year) == 2007 or self._image_set != 'test':
    #         gt_roidb = self.gt_roidb()
    #         rpn_roidb = self._load_rpn_roidb(gt_roidb)
    #         roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
    #     else:
    #         roidb = self._load_rpn_roidb(None)
    #
    #     return roidb
    #
    # def _load_rpn_roidb(self, gt_roidb):
    #     filename = self.config['rpn_file']
    #     print('loading {}'.format(filename))
    #     assert os.path.exists(filename), \
    #         'rpn data not found at: {}'.format(filename)
    #     with open(filename, 'rb') as f:
    #         box_list = pickle.load(f)
    #     return self.create_roidb_from_box_list(box_list, gt_roidb)
    #
    # def _load_selective_search_roidb(self, gt_roidb):
    #     filename = os.path.abspath(
    #         os.path.join(cfg.DATA_DIR, 'selective_search_data',
    #                      self.name + '.mat'))
    #     assert os.path.exists(filename), \
    #         'Selective search data not found at: {}'.format(filename)
    #     raw_data = sio.loadmat(filename)['boxes'].ravel()
    #
    #     box_list = []
    #     for i in range(raw_data.shape[0]):
    #         boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
    #         keep = unique_boxes(boxes)
    #         boxes = boxes[keep, :]
    #         keep = filter_small_boxes(boxes, self.config['min_size'])
    #         boxes = boxes[keep, :]
    #         box_list.append(boxes)
    #
    #     return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_pascal_annotation(self, index):
        '''
        从Images文件夹中加载名称为index的图片，从Labels文件夹中加载名称为index的xml中的信息
        :param index: 图片的名称，不带后缀
        :return: 包含该图片xml信息的dict
        '''
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml') # xml_path
        tree = ET.parse(filename)
        objs = tree.findall('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16) # 放置所有object的bndbox信息
        gt_classes = np.zeros((num_objs), dtype=np.int32) # 放置该object对应的name信息
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32) # 是否有重叠
        seg_areas = np.zeros((num_objs), dtype=np.float32) # bbox矩形面积，H*W
        ishards = np.zeros((num_objs), dtype=np.int32) # difficult信息

        for ix, obj in enumerate(objs): # 遍历每一个object加载对应信息
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            '''
            x1 = float(bbox.find('xmin').text) - 1 # 这个可能是要像素从0开始
            y1 = float(bbox.find('ymin').text) - 1 # 但一个像素的差异不大，除非是最后的边界的情况
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            '''
            diffc = obj.find('difficult')
            difficult = 0 if diffc == None else int(diffc.text)
            ishards[ix] = difficult

            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            # 此处标注的都是text，所以可以用这个_class_to_ind来获取cls，此处都是1.

            boxes[ix, :] = [x1, y1, x2, y2] # bbox的坐标信息
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0 # 重叠信息所有的都是1？
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1) # seg的面积，矩形面积，H*W

        overlaps = scipy.sparse.csr_matrix(overlaps)  # 转化为致密矩阵

        return {
            'boxes': boxes,# 左右bbox四个坐标组成的list：(num_objs, 4), dtype=np.uint16
            'gt_classes': gt_classes, # 所有bbox的name对应的类别，(num_objs), dtype=np.int32
            'gt_ishard': ishards, # 每个bbox是否是difficult
            'gt_overlaps': overlaps, # 是否有重叠？？
            'flipped': False, # 图像是否翻转，原始图像都是False，后面图像增强是经过翻转，会变为True
            'seg_areas': seg_areas # 所有bbox的面积组成的list
        }

    def _get_comp_id(self):
        '''
        获取comp_id
        :return: comp_id
        '''
        comp_id = (self._comp_id + '_' + self._salt
                   if self.config['use_salt'] else self._comp_id)
        return comp_id

    # def _get_voc_results_file_template(self):
    #     filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
    #     filedir = os.path.join(self._devkit_path, 'results',
    #                            'VOC' + self._year, 'Main')
    #     if not os.path.exists(filedir):
    #         os.makedirs(filedir)
    #     path = os.path.join(filedir, filename)
    #     return path
    #
    # def _write_voc_results_file(self, all_boxes):
    #     for cls_ind, cls in enumerate(self.classes):
    #         if cls == '__background__':
    #             continue
    #         print('Writing {} VOC results file'.format(cls))
    #         filename = self._get_voc_results_file_template().format(cls)
    #         with open(filename, 'wt') as f:
    #             for im_ind, index in enumerate(self.image_index):
    #                 dets = all_boxes[cls_ind][im_ind]
    #                 if dets == []:
    #                     continue
    #                 # the VOCdevkit expects 1-based indices
    #                 for k in range(dets.shape[0]):
    #                     f.write(
    #                         '{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(
    #                             index, dets[k, -1], dets[k, 0] + 1,
    #                             dets[k, 1] + 1, dets[k, 2] + 1,
    #                             dets[k, 3] + 1))
