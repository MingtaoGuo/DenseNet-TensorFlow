# DenseNet-TensorFlow
DenseNet for cifar10

## Introduction
DenseNet is simplely implemented by TensorFlow, the flow chart of DneseNet is shown in follow figure.
![](https://github.com/MingtaoGuo/DenseNet-TensorFlow/blob/master/IMAGES/DenseNet.jpg)
The greatest advantage of DneseNet is high accuracy and low occupancy rate of memory. This code we use the DenseNet of 40 depth for cifar10 classification, when we save the model in .ckpt, the ckpt file just cost about 2~3M, it's very slight.
## How to use the code 
1. Download the cifar10 data
2. unzip the cifar10 data, and put them into the folder 'cifar10'


## Results
|Loss|Training accuracy|Validation accuracy|
|-|-|-|
|![](https://github.com/MingtaoGuo/DenseNet-TensorFlow/blob/master/IMAGES/loss.jpg)|![](https://github.com/MingtaoGuo/DenseNet-TensorFlow/blob/master/IMAGES/training_acc.jpg)|![](https://github.com/MingtaoGuo/DenseNet-TensorFlow/blob/master/IMAGES/validation_acc.jpg)|

This experiment we don't use data augmentation, and just train 100 epoches, It doesn't seem to have converged yet. In original paper, 40 depth DenseNet test error is 7%, which is 3 percentage points lower than this code. Due to the poor device, i don't try to train the DenseNet for 300 epoches like the paper. I will very appreciate if somebody can run the code for 300 epoches.
