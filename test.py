import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os
import cv2

from rgbt.rgbt_models.AINet import AINet
from config import opt
from rgbt.dataset import test_dataset



dataset_path = opt.test_path

#set device for test
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
print('USE GPU:', opt.gpu_id)

#load the model
model = AINet()
#Large epoch size may not generalize well. You can choose a good model to load according to the log file and pth files saved in ('./BBSNet_cpts/') when training.
model.load_state_dict(torch.load('/media/zy/shuju/论文相关/AINet/weight/AINet/AINet.pth'))
model.cuda()
model.eval()

#test
test_mae = []
test_datasets = ['VT821','VT1000','VT5000']



for dataset in test_datasets:
    mae_sum  = 0
    save_path = '/home/zy/PycharmProjects/SOD/rgbt/rgbt_test_maps/CESHI/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/RGB/'
    gt_root = dataset_path + dataset + '/GT/'
    ti_root=dataset_path + dataset +'/T/'
    test_loader = test_dataset(image_root, gt_root,ti_root, opt.testsize)
    for i in range(test_loader.size):
        image, gt, ti, name  = test_loader.load_data()
        gt = gt.cuda()
        # print(gt.type())
        image = image.cuda()
        ti = ti.cuda()
        res  = model(image,ti)
        predict = torch.sigmoid(res)
        predict = (predict - predict.min()) / (predict.max() - predict.min() + 1e-8)
        mae = torch.sum(torch.abs(predict - gt)) / torch.numel(gt)
        mae_sum = mae.item() + mae_sum
        predict = predict.data.cpu().numpy().squeeze()
        # print(predict.shape)
        print('save img to: ',save_path+name)
        cv2.imwrite(save_path+name, predict*255)
    test_mae.append(mae_sum / test_loader.size)
print('Test Done!', 'MAE', test_mae)
