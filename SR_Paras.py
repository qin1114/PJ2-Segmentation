"""
训练SRNN网络，获得参数SR
Train:
    Input: Images: Gold Standard: binary; batch_size*type_num*w*h
           Labels: Gold Standard: [0,1,2,3]; batch_size*w*h
"""
import torch
import os
import time
import sys
import numpy as np

# BASE_DIR = os.path.dirname(os.path.abspath("__file__")) #in jupyter
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #in py:/home/liujiyao/qmh-PJ2
sys.path.append(BASE_DIR)

from core.DataLoader import NPYSegDataset as dataSet
from core.TrainFunction import Train, try_all_gpus, Animator, write_line
from core.Net import SRNN, UNet

#Global variables
os.environ["CUDA_VISIBLE_DEVICE"]="1"
save_foldername = os.path.join(os.getcwd(), "Result") #path to save intermediate results
# txt_filename = "print_SRParasMessage_version1.txt"
# png_filename = "SRParas_version1.png"
# tar_filename = "savePara_SR_epo30_LE_CV6_3LEGtypes.tar"
txt_filename = "temp_L2_CV1_version2.txt"
png_filename = "temp_L2_CV1_version2.png"
tar_filename = "temp_L2_CV1_version2.tar"


batch_size = 32
net = SRNN()
num_epochs, lr, wd, devices = 30, 0.001, 5e-3, torch.device("cuda:1")
trainer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1], nrows=3, ncols=2,
                                legend=['train loss', 'train acc', 'test acc'])
row_index = [0,0,1,1,2,2]
col_index = [0,1,0,1,0,1]

time_list = []
time_list.append(time.time()) #Record first time
for kNum in range(1): #Only One CrossValid to avoid to be all 0s
      """Train net parameters"""
      trainData = dataSet(is_train=True, kNum=kNum, subType=["DE","C0","T2"])
      testData = dataSet(is_train=False, kNum=kNum, subType=["DE","C0","T2"])
      train_iter = torch.utils.data.DataLoader(trainData, batch_size, shuffle=True, drop_last=True) #由于剪裁结果一定相等，不需要指定collate_fn
      test_iter = torch.utils.data.DataLoader(testData, batch_size, shuffle=True, drop_last=True) #由于剪裁结果一定相等，不需要指定collate_fn

      Train(net, train_iter=train_iter, test_iter=test_iter,
            loss_type="weighted_cross_entropy+generalized_dice_loss", regular_type=None,
            use_SC=False, is_pretrain_SR=True,
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




