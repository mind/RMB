CUDA_VISIBLE_DEVICES=1 python demo.py \
--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
--image_folder /home/dingzy/TorchModel_new/jue_5/ \
--saved_model saved_models/TPS-ResNet-BiLSTM-CTC-Seed1111/best_accuracy.pth
