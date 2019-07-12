运行环境：
pytorch、sklearn、albumentations、cv2、numpy、pandas

以下是运行说明，大概需要运行1小时左右。

cd multi-digit-pytorch/
python 2_predict.py

cd ../crnn-pytorch/
python test2_tta.py --snapshot tmp/crnn_resnet18_0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_best --visualize False --data-path ../data/
python submit.py

生成的`tmp_rcnn_tta10_pb_submit.csv`就是最终的提交文件。