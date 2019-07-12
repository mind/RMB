1. install the dependency packages
pip3 install torch==1.1.0
pip3 install lmdb pillow torchvision nltk natsort

2. use the faster rcnn to crop the ROI images
1.首先安装mmdet，pytorch为1.1.0版 github地址为：https://github.com/open-mmlab/mmdetection
2.python test.py，启动预测脚本
3.得到预测结果，python pick_picname.py 启动切片脚本，进行切片
4.切片结束，完成检测任务和训练数据准备任务


3. 
cd deep-text-recognition-benchmark-master
Train the models
CRNN: 
	CUDA_VISIBLE_DEVICES=0 python3 train.py \
--train_data data_lmdb_release/training --valid_data data_lmdb_release/validation \
--select_data MJ-ST --batch_ratio 0.5-0.5 \
--Transformation None --FeatureExtraction VGG --SequenceModeling BiLSTM --Prediction CTC

ATTENTION: 
	CUDA_VISIBLE_DEVICES=0 python3 train.py \
--train_data data_lmdb_release/training --valid_data data_lmdb_release/validation \
--select_data MJ-ST --batch_ratio 0.5-0.5 \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn


4. Prediction
use the CTC model:
CUDA_VISIBLE_DEVICES=0 python3 demo.py \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
--image_folder ****(DIY_PATH) \
--saved_model ./saved_models/TPS-ResNet-BiLSTM-CTC-Seed1111/best_accuracy.pth


CUDA_VISIBLE_DEVICES=0 python3 demo.py \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
--image_folder ****(DIY_PATH)/ \
--saved_model ./saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/best_accuracy.pth

在这次比赛中，我们发现CTC比ATTN的acc更加的高，所以我们采用的是CTC。

other: When you need to create lmdb dataset
pip3 install fire
python3 create_lmdb_dataset.py --inputPath data/ --gtFile data/gt.txt --outputPath result/

