## 第一部分： 编码切割
text-detection-ctpn： 编码位置识别代码， 主要来源于github：https://github.com/eragonruan/text-detection-ctpn
将最后的输入内容应用到了图像上， 进行了切图。
手动标记训练数据1000张， 训练的log日志， 文件夹中有

## 第二部分： 编码识别
代码： CRNN.ipynb
编码识别主要采用CNN + RNN + CTC
分别采用不同的CNN网络去提取图片特征， 进行训练， 最后进行预测

## 第三部分： 结果融合采用了4个主干网络， 对预测的60中结果进行投票融合， 得到最终结果
