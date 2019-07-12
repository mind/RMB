###所有文件都可以在google colab直接打开运行
检测部分：
1. 下载ctpn代码及模型，地址
 https://drive.google.com/drive/folders/1VdsIQ4sgGmNI8Gor-_ie3xktLMFDp8Sd?usp=sharing
 因为包含vgg16和ctpn的模型，较大

2.运行 tinymind.ipynb 中ctpn部分代码，生成剪切图片 （训练数据为人工标记160多张编码位置）
  备注：99%只检测到一个候选框，可以直接用
  剩余部分需要简单过滤，过滤完仍有多个框的全部送到检测，对结果进行正则判断，错误框都是一堆乱的数字，极少（大概三张）会出现边缘少了一点，预测结果正则会检测出来，运行demo_1重新截取

识别部分：
crnn 模型地址：
运行代码在 tinymind.ipynb
https://drive.google.com/file/d/1ywyH25xtcSHhZxeIACgso4Bslo-ZidNV/view?usp=sharing
代码在densent.ipynb
densent模型地址：
https://drive.google.com/file/d/1_xU2d7bU6FOLHJPjy1dgDLitdlGHCS7-/view?usp=sharing
代码在ASTER.ipynb
aster模型地址:
https://drive.google.com/drive/folders/1ctd55IG30aRAC4xUcSyirOyKaWW7mAfh?usp=sharing
experiments 解压到 aster文件夹下面

模型融合代码在 tinymind.ipynb 最后部分

2D9QKGRM.jpg 为残缺图片,强制覆盖的结果
