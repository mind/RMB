# -*- coding: UTF-8 -*-

"""
# version: python3.6
# author: TechDing
# email: dingustb@126.com
# file: model_predict.py
# time: 2019/7/2 10:54
# doc: 使用成熟的CTC_Models来预测roi_setX.npy，并将结果保存
"""

import string
CHARS = string.digits + string.ascii_uppercase

import numpy as np
import keras.backend as K
def pred_imgs(base_model,setX):
    '''
    使用base_model来预测roi图片
    :param base_model: 已经训练好的model
    :param setX: roi图片组成的数据，必须为0-255， uint8,shape为（bs,80,240,3）
    :return:
    '''
    def predict_batch(batchX):
        testX2=batchX.astype('float64')/255
        testX2=testX2.transpose(0,2,1,3)
        y_pred=base_model.predict(testX2)
        y_pred = y_pred[:,2:,:] # (bs, 24,37)
        out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0])*y_pred.shape[1], )[0][0])[:, :10]
        return [''.join([CHARS[i] for i in line]) for line in out]

    all_results=[]
    if len(setX)>1000:
        for i in range(0,len(setX),1000):
            end=min(len(setX),i+1000)
            batchX=setX[i:end]
            all_results.extend(predict_batch(batchX))
    else:
        all_results=predict_batch(setX)
    return all_results

from keras.models import load_model
import pandas as pd
import os
def ctc_model_predict(model_type,test_roiX_path,test_names_path,result_save_path):
    '''
    使用成熟的ctc模型来预测roi图片，该roi图片以npy的形式保存到test_roiX_path中，图片对应的名称为test_names_path中
    :param model_type: 所使用的模型类型，必须为{cv,ctpn1,ctpn2}三者之一
    :param test_roiX_path: roi图片，以npy的形式保存，0-255，uint8
    :param test_names_path: 图片名称保存的路径
    :param result_save_path: 最终预测的结果保存的路径，该结果为df，columns=['name', 'label']
    :return:
    '''
    # print('start to predict roi imgs by ctc_model')
    models_dict={'cv':os.path.abspath('./CTC_Models/Models/CV_Model.h5'),
                 'ctpn1':os.path.abspath('./CTC_Models/Models/FTPN_Model1.h5'),
                 'ctpn2':os.path.abspath('./CTC_Models/Models/FTPN_Model2.h5')}
    ctc_model_path=models_dict.get(model_type,'Error model path')
    testX=np.load(test_roiX_path)
    img_names=np.load(test_names_path)
    ctc_model = load_model(ctc_model_path)  # 模型加载非常耗时，需要注意
    predicted=pred_imgs(ctc_model,testX)
    result=np.c_[img_names,np.array(predicted)]
    df = pd.DataFrame(result, columns=['name', 'label'])
    df.to_csv(result_save_path,index=False)
    print('predicted result is saved to {}'.format(result_save_path))
