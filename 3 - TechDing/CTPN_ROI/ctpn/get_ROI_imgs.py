from __future__ import print_function
import tensorflow as tf
import numpy as np
import os, sys, cv2
sys.path.append('../') # E:\PyProjects\Codes\RMB_TechDing\CTPN_ROI\
from .lib import get_network,cfg,cfg_from_file,test_ctpn,TextDetector
from .lib.text_connector.text_connect_cfg import Config as TextLineCfg

def resize_im(im, scale, max_scale=None):
    '''
    将原始图像im调整到一定大小，
    :param im: 原始图像
    :param scale: 最小比例：eg 600
    :param max_scale: 最大比例： eg 1200
    :return: 调整之后的图像，调整比例
    '''
    f=float(scale)/min(im.shape[0], im.shape[1])
    if max_scale!=None and f*max(im.shape[0], im.shape[1])>max_scale:
        f=float(max_scale)/max(im.shape[0], im.shape[1])
    return cv2.resize(im, None,None, fx=f, fy=f,interpolation=cv2.INTER_LINEAR), f

def get_save_roi(img,image_name,boxes,scale,save_rois_folder):
    '''
    提取出图像img中的有效ROI图片，并将该ROI图片保存到save_rois_folder中
    :param img:
    :param image_name: 图像的名称，保存的ROI图片以该名称命名
    :param boxes: ctpn模型得到的候选框
    :param scale: 图像缩放时所用的比例
    :param save_rois_folder: 该ROI图片都会保存到该文件夹中
    :return:
    '''
    # 将图片按照比例缩放
    img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)
    base_name = image_name.split(os.sep)[-1]
    # 将坐标信息还原到原图大小，然后提取出roi。
    box_num=0
    xmin,ymin,xmax,ymax=0,0,0,0
    for box in boxes:
        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5: # 排除长度太小的框
            continue
        # 根据resize所用的比例，将box的坐标还原到原图上的坐标信息
        min_x = min(int(box[0]/scale),int(box[2]/scale),int(box[4]/scale),int(box[6]/scale))
        min_y = min(int(box[1]/scale),int(box[3]/scale),int(box[5]/scale),int(box[7]/scale))
        max_x = max(int(box[0]/scale),int(box[2]/scale),int(box[4]/scale),int(box[6]/scale))
        max_y = max(int(box[1]/scale),int(box[3]/scale),int(box[5]/scale),int(box[7]/scale))
        # 比例调整，W/H比例必须>=6 # 防止框太窄
        W0 = max_x - min_x
        H0 = max_y - min_y
        ratio0 = float(W0 / H0)
        newXmin,newXmax=0,0
        if ratio0 < 6.0:  # need to enlarge
            newW = 6.0 * H0
            halfW = int((newW - W0) / 2)
            newXmin = min_x - halfW
            newXmax = max_x + halfW
            real_W=img.shape[1]
            if newXmax >= real_W:
                newXmin = real_W - newW - 1
                newXmax = real_W - 1
        box_num=1
        xmin=int(newXmin) if newXmin>0 else min_x
        xmax=int(newXmax-1) if newXmax>0 else max_x-1
        ymin=int(min_y)
        ymax=int(max_y-1)
        break # 只取第一个box即可
    if box_num!=1: # 如果==0，说明长度太小，不是正确的ROI，只能print提示一下
        # print('Error: found {} objects in {}'.format(box_num,image_name))
        pass
    else:
        cv2.imwrite(os.path.join(save_rois_folder,base_name), img[ymin:ymax,xmin:xmax])
        # roi保存到save_rois_folder

def ctpn(sess, net, image_name,save_rois_folder):
    '''
    使用成熟的CTPN网络来预测图像image_name，并将预测得到的ROI图片保存到save_rois_folder中
    :param sess: tensorflow的sess
    :param net:  模型的网络结构
    :param image_name: 图像路径
    :param save_rois_folder: ROI保存的文件夹
    :return:
    '''
    img = cv2.imread(image_name)
    img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    scores, boxes = test_ctpn(sess, net, img) # 使用ctpn网络对图img进行预测，得到scores分数和boxes坐标信息
    # scores是所有proposals的概率，从大到小排列
    # boxes 是每个proposal的坐标，四个值，xmin,ymin,xmax,ymax
    textdetector = TextDetector() # 使用textdetector进行box的进一步处理
    boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    if boxes.shape[0]==0: # 如果处理之后的候选框为0，表示没有找到这个box
        # print('Error: No box found in {}'.format(image_name))
        pass
    else: # 如果有多个候选框，继续缩小范围，并保存roi图片
        get_save_roi(img, image_name, boxes, scale,save_rois_folder)

from glob import glob
def get_save_ctpn_roi(src_imgs_folder,save_rois_folder):
    '''
    使用模型ckpt来提取出图片中的rois，并将所有的rois保存到save_rois_folder中
    :param src_imgs_folder:原始的，需要提取roi的图片文件夹
    :param save_rois_folder: 提取的roi需要保存的文件夹
    :return:
    '''
    if not os.path.exists(save_rois_folder):
        os.makedirs(save_rois_folder)

    cfg_from_file(os.path.abspath('./CTPN_ROI/ctpn/text.yml')) # 加载预测所用的配置
    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    # load network
    net = get_network("VGGnet_test") # 加载VGGnet_test网络
    # load model
    print(('Loading network {:s}... '.format("VGGnet_test")))
    saver = tf.train.Saver()
    ckpt_path=os.path.abspath('./CTPN_ROI/checkpoints')
    try:
        print('ckpt path: ', ckpt_path)
        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
        saver.restore(sess, ckpt.model_checkpoint_path)
    except:
        raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in range(10):  # 对网络模型进行预热
        _, _ = test_ctpn(sess, net, im)
    im_names = glob(os.path.join(src_imgs_folder, '*.jpg')) # 需要预测的所有图片路径
    for idx, im_name in enumerate(im_names):
        ctpn(sess, net, im_name, save_rois_folder)
        print('\r{}/{} finished...'.format(idx + 1, len(im_names)), end=' ')

# if __name__ == '__main__':
    # sys.path.append('./')
    # sys.path.append('./lib')
    # src_imgs_folder='/home/ray/Codes/RMB_TechDing/TestImgs'
    # save_rois_folder='/home/ray/Codes/RMB_TechDing/TEMP/ctpn_roi_imgs_test'
    # get_save_ctpn_roi(src_imgs_folder, save_rois_folder)
