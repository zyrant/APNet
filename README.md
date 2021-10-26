# APNet
Code and result about APNet(IEEE TETCI)<br>
'APNet: Adversarial-Learning-Assistance and Perceived Importance Fusion Network for All-Day RGB-T Salient Object Detection' 
![image](https://user-images.githubusercontent.com/38373305/134764453-4db0e79f-77f2-448f-a32d-76c907fff0aa.png)

# Requirements
Python 3.7, Pytorch 1.5.0+, Cuda 10.2, TensorboardX 2.1, opencv-python

# Dataset and Evaluate tools
RGB-T SOD Datasets can be found in:  https://github.com/lz118/RGBT-Salient-Object-Detection <br>

Evaluate tools: we use the matlab verison provide by [Dengping Fan](http://dpfan.net/d3netbenchmark/).

# Result
![image](https://user-images.githubusercontent.com/38373305/134769028-f40316ec-b586-4064-aaf0-bffc4b34d18f.png)

Test saliency maps in all datasets[predict]:  [baidu](https://pan.baidu.com/s/1bmlNxOvZkaiwc4EwqY1Nlw)  提取码：vy3r <br>

The pretrained model can be downloaded at[APNet.pth]:  [baidu](https://pan.baidu.com/s/1bmlNxOvZkaiwc4EwqY1Nlw)  提取码：vy3r <br>

PS: we resize the testing data to the size of 224 * 224 for quicky evaluate[GT for matlab], [baidu](https://pan.baidu.com/s/1bmlNxOvZkaiwc4EwqY1Nlw)  提取码：vy3r <br>

# Citation
@ARTICLE{9583676,  author={Zhou, Wujie and Zhu, Yun and Lei, Jingsheng and Wan, Jian and Yu, Lu},  <br> journal={IEEE Transactions on Emerging Topics in Computational Intelligence},  <br>  title={APNet: Adversarial Learning Assistance and Perceived Importance Fusion Network for All-Day RGB-T Salient Object Detection},  <br>  year={2021},  volume={},  <br> number={},  <br> pages={1-12},  <br> doi={10.1109/TETCI.2021.3118043}}
# Acknowledgement
The implement of this project is based on the code of ‘Cascaded Partial Decoder for Fast and Accurate Salient Object Detection, CVPR2019’and 'BBS-Net: RGB-D Salient Object Detection with a Bifurcated Backbone Strategy Network' proposed by Wu et al and Deng et al.

# Contact
Please drop me an email for further problems or discussion: zzzyylink@gmail.com or wujiezhou@163.com
