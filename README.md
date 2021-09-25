# APNet
Code and result about APFNet(IEEE TETCI)<br>
'APNet: Adversarial-Learning-Assistance and Perceived Importance Fusion Network for All-Day RGB-T Salient Object Detection' 
![image](https://user-images.githubusercontent.com/38373305/134764423-e5d02a14-3b7f-4d85-9fdc-36502b6fc951.png)

# Requirements
Python 3.7, Pytorch 1.5.0+, Cuda 10.2, TensorboardX 2.1, opencv-python

# Dataset and Evaluate tools
RGB-D SOD Datasets can be found in:  http://dpfan.net/d3netbenchmark/  or https://github.com/jiwei0921/RGBD-SOD-datasets <br>

we use the matlab verison provide by Dengping Fan, and we provide our test datesets [百度网盘](https://pan.baidu.com/s/1tVJCWRwqIoZQ3KAplMSHsA) 提取码：zust 

# Result
Test maps: [百度网盘](https://pan.baidu.com/s/1QcEAHlS8llyX-i3kX4npAA)  提取码：zust <br>
Pretrained model download:[百度网盘](https://pan.baidu.com/s/1reGFvIYX7rZjzKuaDcs-3A)  提取码：zust <br>
PS: we resize the testing data to the size of 224 * 224 for quicky evaluate, [百度网盘](https://pan.baidu.com/s/1t5cES-RAnMCLJ76s9bwzmA)  提取码：zust <br>

# Citation
@ARTICLE{9424966,<br>
  author={Zhou, Wujie and Zhu, Yun and Lei, Jingsheng and Wan, Jian and Yu, Lu},<br>
  journal={IEEE Transactions on Multimedia}, <br>
  title={CCAFNet: Crossflow and Cross-scale Adaptive Fusion Network for Detecting Salient Objects in RGB-D Images}, <br>
  year={2021},<br>
  doi={10.1109/TMM.2021.3077767}}<br>

# Acknowledgement
The implement of this project is based on the code of ‘Cascaded Partial Decoder for Fast and Accurate Salient Object Detection, CVPR2019’and 'BBS-Net: RGB-D Salient Object Detection with a Bifurcated Backbone Strategy Network' proposed by Wu et al and Deng et al.

# Contact
Please drop me an email for further problems or discussion: zzzyylink@gmail.com or wujiezhou@163.com