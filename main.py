"""
@author: Menghan Qin
Combinate all networks training: Unet; SCNN; SRNN; SRSCN
Changable parameters:
    :param SR_path: path to load pretrained SR-Net parameters
    :param use_SR: whether to use SR regular-term
    :param is_pretrain_SR: whether to pretrain SR-Net(to be False all the time)
    :param regular_type: Str: name of regular-term
    :param use_SC: whether to use SC regular-term
    :param out_SC: middle output of SC-Net(generated in Train function, needed in Loss-term)
set optimizer: Adam;    loss_funtion: weighted_cross_entropy+generalized_dice_loss
    accuracy_funtion: ["dice", "Average Symmetric Surface Diatance","Hausdorff distance",
                      "correct prediction of pixel", "specificity", "sensitivity", "precision", "accuracy"]
    accuracy shown in figure: dice

Main Net: Unet(all change based on main network: Unet), get an input: use_SC
[1] Unet
SR_path = None;  use_SR = False;  is_pretrain_SR = False;  regular_type = None;  use_SC = None
[2] SRNN
SR_path = "path";  use_SR = True;  is_pretrain_SR = False;  regular_type = "SR";  use_SC = False
(I tried 2 parameters of pretrained net, with weight_decay to be 1e-3 and 5e-3 respectively)
[3] SCNN(batch_size to be 8 to deal with "CUDA out of range Problem" caused by SC regular-term)
SR_path = None;  use_SR = False;  is_pretrain_SR = False;  regular_type = "SC";  use_SC = True
[4] SCSRN(batch_size = 8)
SR_path = "path";  use_SR = True;  is_pretrain_SR = False;  regular_type = "SC+SR";  use_SC = True
"""

#SCSRN

import torch
import os
import time
import sys
import numpy as np

# BASE_DIR = os.path.dirname(os.path.abspath("__file__")) #in jupyter
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #in py:/home/liujiyao/qmh-PJ2
sys.path.append(BASE_DIR)

from core.LossFunction import CrossCompose as loss_fun
from core.AccuracyMetrix import AccuracyCompose as accuracy
from core.DataLoader import NPYSegDataset as dataSet
from core.TrainFunction import Train, try_all_gpus, Animator, write_line
from core.Net import FCNNet, UNet, SRNN

#Global variables
os.environ["CUDA_VISIBLE_DEVICE"]="1"
save_foldername = os.path.join(os.getcwd(), "Result") #path to save intermediate results
# SRNN_version1_L2CV1
# txt_filename = "print_SRNN_message_Adam_L2CV1.txt"
# png_filename = "SRNN_Adam_L2CV1.png"
# tar_filename = "save_SRNN_epo30_0.7GD+0.3WL_CV6_3LEGtypes_Adam_L2CV1.tar"
# SR_path = "/home/liujiyao/qmh-PJ2/core/Result/save_SRParas_epo30_0.7GD+0.3WL_CV1_3LEGtypes.tar"
# SRNN_version2_CV1
# txt_filename = "print_SRNN_message_Adam_CV1.txt"
# png_filename = "SRNN_Adam_CV1.png"
# tar_filename = "save_SRNN_epo30_0.7GD+0.3WL_CV6_3LEGtypes_Adam_CV1.tar"
# use_SC = False
# use_SR = True
# is_pretrain_SR = False
# regular_type = "SR"

# SCNN
# txt_filename = "print_SCNN_message__version1.txt"
# png_filename = "SCNN_version1.png"
# tar_filename = "save_SCNN_epo30_0.7GD+0.3WL_CV6_3LEGtypes.tar"
# use_SC = True
# use_SR = False
# is_pretrain_SR = False
# regular_type = "SC"

# UNet
# txt_filename = "print_UNet_message_version1.txt"
# png_filename = "UNet_version1.png"
# tar_filename = "save_UNet_epo30_0.7GD+0.3WL_CV6_3LEGtypes.tar"

"""SCSRN"""
txt_filename = "print_SCSRN_message_Adam_CV1_3C7R.txt"
png_filename = "SCSRN_Adam_CV1_3C7R.png"
tar_filename = "save_SCSRN_epo30_0.7GD+0.3WL_CV6_3LEGtypes_Adam_CV1_3C7R.tar"
SR_path = "/home/liujiyao/qmh-PJ2/core/Result/save_SRParas_epo30_0.7GD+0.3WL_L2CV1_3LEGtypes.tar"
use_SC = True
use_SR = True
is_pretrain_SR = False
regular_type = "SR+SC"


batch_size = 8
net = UNet(use_SC=use_SC)
num_epochs, lr, wd, devices = 30, 0.001, 1e-3, torch.device("cuda:1")
trainer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
# trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1], nrows=3, ncols=2,
                                legend=['train loss', 'train acc', 'test acc'])
row_index = [0,0,1,1,2,2]
col_index = [0,1,0,1,0,1]

time_list = []
time_list.append(time.time()) #Record first time
for kNum in range(6):
      """Train net parameters"""
      trainData = dataSet(is_train=True, kNum=kNum, subType=["DE","C0","T2"])
      testData = dataSet(is_train=False, kNum=kNum, subType=["DE","C0","T2"])
      train_iter = torch.utils.data.DataLoader(trainData, batch_size, shuffle=True, drop_last=True) #由于剪裁结果一定相等，不需要指定collate_fn
      test_iter = torch.utils.data.DataLoader(testData, batch_size, shuffle=True, drop_last=True) #由于剪裁结果一定相等，不需要指定collate_fn

      Train(net, train_iter=train_iter, test_iter=test_iter,
            loss_type="weighted_cross_entropy+generalized_dice_loss",
            use_SC=use_SC, SR_path=SR_path, regular_type=regular_type, is_pretrain_SR=False, use_SR=use_SR,
            trainer=trainer, num_epochs=num_epochs, devices=devices,
            acc_type=["dice", "Average Symmetric Surface Diatance","Hausdorff distance",
                      "correct prediction of pixel", "specificity", "sensitivity",
                      "precision", "accuracy"], animator_type = "dice",
            kNum=kNum, Animator=animator, row=row_index[kNum], col=col_index[kNum],
            foldername=save_foldername, filename=txt_filename).train() #Object and train model
      time_list.append(time.time())


for index in range(len(time_list)-1):
    str1 = 'kNum: %d; time cost: %.3f s'%(index, time_list[index+1]-time_list[index])
    print(str1)
    write_line(foldername=save_foldername, filename=txt_filename, str=str1)

torch.save(net.state_dict(), os.path.join(save_foldername, tar_filename))

for row in range(3):
    for col in range(2):
        animator.show_axes(foldername=save_foldername, filename=png_filename,
                           row=row, col=col)