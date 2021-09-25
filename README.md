# APNet
Code and result about APFNet(IEEE TETCI)<br>
'APNet: Adversarial-Learning-Assistance and Perceived Importance Fusion Network for All-Day RGB-T Salient Object Detection' 
![image](https://user-images.githubusercontent.com/38373305/134764453-4db0e79f-77f2-448f-a32d-76c907fff0aa.png)

<center class = "half">
<img src = “https://user-images.githubusercontent.com/38373305/134768650-5b0b40fc-9319-49c4-a283-720c9ec3bf35.png”  width = “10%” align = left><img src = “https://user-images.githubusercontent.com/38373305/134768658-532c4cff-44bd-42ab-8d5a-bd62ee4c2060.png”  width = “10%” align = right>


![image](https://user-images.githubusercontent.com/38373305/134768670-ef364f98-bb95-4d2f-b794-81580d268c30.png)


# Requirements
Python 3.7, Pytorch 1.5.0+, Cuda 10.2, TensorboardX 2.1, opencv-python

# Dataset and Evaluate tools
RGB-T SOD Datasets can be found in:  https://github.com/lz118/RGBT-Salient-Object-Detection <br>

Evaluate tools: we use the matlab verison provide by [Dengping Fan](http://dpfan.net/d3netbenchmark/).

# Result
Test saliency maps in all datasets[predict]:  [baidu](https://pan.baidu.com/s/1bmlNxOvZkaiwc4EwqY1Nlw)  提取码：vy3r <br>

The pretrained model can be downloaded at[APNet.pth]:  [baidu](https://pan.baidu.com/s/1bmlNxOvZkaiwc4EwqY1Nlw)  提取码：vy3r <br>

PS: we resize the testing data to the size of 224 * 224 for quicky evaluate[GT for matlab], [baidu](https://pan.baidu.com/s/1bmlNxOvZkaiwc4EwqY1Nlw)  提取码：vy3r <br>

# Citation

# Acknowledgement
The implement of this project is based on the code of ‘Cascaded Partial Decoder for Fast and Accurate Salient Object Detection, CVPR2019’and 'BBS-Net: RGB-D Salient Object Detection with a Bifurcated Backbone Strategy Network' proposed by Wu et al and Deng et al.

# Contact
Please drop me an email for further problems or discussion: zzzyylink@gmail.com or wujiezhou@163.com
