from mmdet.apis import init_detector, inference_detector, show_result
import os
import cv2

#配置文件
config_file = './faster_rcnn_r50_fpn_1x_voc0712.py'
#训练模型的地址
checkpoint_file = './faster_rcnn_r50_fpn_1x_voc0712/epoch_4.pth'
#记录坐标的txt文件的地址
f = open('./result_rcnn.txt','w')

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
#待预测图片文件夹的地址
for root,dirs,files in os.walk('../private_test_data'):
    for file in files:
        pic_path = os.path.join(root,file)
        img = cv2.imread(pic_path)
        result = inference_detector(model, img)
        print(result[0][0])
        try:
            length = len(result[0][0])
        except IndexError:
            f.write(file)
            f.write('\n')
            continue       
        f.write(file)
        f.write(' ')
        f.write(str(int(result[0][0][0])))
        f.write(',')
        f.write(str(int(result[0][0][1])))
        f.write(',')
        f.write(str(int(result[0][0][2])))
        f.write(',')
        f.write(str(int(result[0][0][3])))
        f.write('\n')
        f.flush()

