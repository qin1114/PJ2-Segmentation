import torch
from core.Net import FCNNet, UNet
import os
import numpy as np
import torch.nn.functional as F
import sys
import time
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #in py:/home/liujiyao/qmh-PJ2
sys.path.append(BASE_DIR)

from core.DataLoader import NPYSegDataset as dataSet
from core.AccuracyMetrix import AccuracyCompose as accuracy


# os.environ["CUDA_VISIBLE_DEVICE"]="0"
# device = "cuda:0"
device = "cpu"
start_time = time.time()

# UNet
use_SC = False
net = UNet(use_SC=use_SC).to(device).eval() #Set net to be evaluate mode
net_name = "UNet"
net.load_state_dict(torch.load(os.path.join(os.getcwd(), "Result", net_name,
                            "save_UNet_epo30_0.7GD+0.3WL_CV6_3LEGtypes_Adam.tar")
                    , map_location=torch.device('cpu')))
# net.load_state_dict(torch.load(os.path.join(os.getcwd(), "Result", net_name,
#                             "save_UNet_epo30_0.7GD+0.3WL_CV6_3LEGtypes_Adam.tar")))

# UNet_SGD
use_SC = False
net_SGD = UNet(use_SC=use_SC).to(device).eval() #Set net to be evaluate mode
net_name = "UNet"
net_SGD.load_state_dict(torch.load(os.path.join(os.getcwd(), "Result", net_name,
                            "save_Unet_epo30_0.7GD+0.3WL_CV6_3LEGtypes_SGD.tar")
                    , map_location=torch.device('cpu')))

# SCNN_V1
# use_SC = True
# net_SCNN = UNet(use_SC=use_SC).to(device).eval()
# net_name = "SCNN"
# net_SCNN.load_state_dict(torch.load(os.path.join(os.getcwd(), "Result", net_name,
#                             "save_SCNN_epo30_0.7GD+0.3WL_CV6_3LEGtypes.tar")
#                          , map_location=torch.device('cpu')))
# net_SCNN.load_state_dict(torch.load(os.path.join(os.getcwd(), "Result", net_name,
#                             "save_SCNN_epo30_0.7GD+0.3WL_CV6_3LEGtypes.tar")))

# SCNN_V2
use_SC = True
net_SCNN_V2 = UNet(use_SC=use_SC).to(device).eval()
net_name = "SCNN"
net_SCNN_V2.load_state_dict(torch.load(os.path.join(os.getcwd(), "Result", net_name,
                            "save_SCNN_epo30_0.7GD+0.3WL_CV6_3LEGtypes_V2.tar")
                         , map_location=torch.device('cpu')))

# SRNN_L2
use_SC = False
net_SRNN_L2 = UNet(use_SC=use_SC).to(device).eval()
net_name = "SRNN"
net_SRNN_L2.load_state_dict(torch.load(os.path.join(os.getcwd(), "Result", net_name,
                            "save_SRNN_epo30_0.7GD+0.3WL_CV6_3LEGtypes_Adam_L2CV1.tar")
                            , map_location=torch.device('cpu')))
# net_SRNN_L2.load_state_dict(torch.load(os.path.join(os.getcwd(), "Result", net_name,
#                             "save_SRNN_epo30_0.7GD+0.3WL_CV6_3LEGtypes_Adam_L2CV1.tar")))

# SRNN
use_SC = False
net_SRNN = UNet(use_SC=use_SC).to(device).eval()
net_name = "SRNN"
net_SRNN.load_state_dict(torch.load(os.path.join(os.getcwd(), "Result", net_name,
                            "save_SRNN_epo30_0.7GD+0.3WL_CV6_3LEGtypes_Adam_CV1_4e5.tar")
                         , map_location=torch.device('cpu')))
# net_SRNN.load_state_dict(torch.load(os.path.join(os.getcwd(), "Result", net_name,
#                             "save_SRNN_epo30_0.7GD+0.3WL_CV6_3LEGtypes_Adam_L2CV1_4e5.tar")))

# SCSRN: 0.7SC+0.3SR
use_SC = True
net_SCSRN_3C7R = UNet(use_SC=use_SC).to(device).eval()
net_name = "SCSRN"
net_SCSRN_3C7R.load_state_dict(torch.load(os.path.join(os.getcwd(), "Result", net_name,
                            "save_SCSRN_epo30_0.7GD+0.3WL_CV6_3LEGtypes_Adam_CV1_3C7R.tar")
                               , map_location=torch.device('cpu')))
# net_SCSRN_3C7R.load_state_dict(torch.load(os.path.join(os.getcwd(), "Result", net_name,
#                             "save_SCSRN_epo30_0.7GD+0.3WL_CV6_3LEGtypes_Adam_CV1_3C7R.tar")))

# SCSRN: 0.5SC+0.5SR
# use_SC = True
# net_SCSRN_5C5R = UNet(use_SC=use_SC).to(device).eval()
# net_name = "SCSRN"
# net_SCSRN_5C5R.load_state_dict(torch.load(os.path.join(os.getcwd(), "Result", net_name,
#                             "save_SCSRN_epo30_0.7GD+0.3WL_CV6_3LEGtypes_Adam_CV1_5C5R.tar")
#                                , map_location=torch.device('cpu')))
# net_SCSRN_8C3R.load_state_dict(torch.load(os.path.join(os.getcwd(), "Result", net_name,
#                             "save_SCSRN_epo30_0.7GD+0.3WL_CV6_3LEGtypes_Adam_CV1_8C3R.tar")))

# SCSRN: 0.3SC+1SR
# use_SC = True
# net_SCSRN_3C10R = UNet(use_SC=use_SC).to(device).eval()
# net_name = "SCSRN"
# net_SCSRN_3C10R.load_state_dict(torch.load(os.path.join(os.getcwd(), "Result", net_name,
#                             "save_SCSRN_epo30_0.7GD+0.3WL_CV6_3LEGtypes_Adam_CV1_3C10R.tar")
#                                , map_location=torch.device('cpu')))


# SCSRN: 1SC+1SR
use_SC = True
net_SCSRN_10C10R = UNet(use_SC=use_SC).to(device).eval()
net_name = "SCSRN"
net_SCSRN_10C10R.load_state_dict(torch.load(os.path.join(os.getcwd(), "Result", net_name,
                            "save_SCSRN_epo30_0.7GD+0.3WL_CV6_3LEGtypes_Adam_CV1_10C10R.tar")
                               , map_location=torch.device('cpu')))


def pngToRGB(img_array, nrows=1, ncols=2, figure_size=(10,6), sub_title=None,
             save_folderpath=os.path.join(os.getcwd(),"Result", "ShowImages"),
             save_filepath="subject10_DE_5.png"):
    # Input: img_array: image_num * type_num*w*h, show it with given nrows and ncols
    # 将标签变为彩色图片进行存储
    savetemp = np.zeros((img_array.shape[0], img_array[0].shape[0], img_array[0].shape[1], 3))  # 用于存储彩色图片

    for img_index in range(img_array.shape[0]):
        # 布尔值矩阵（第一维B，第二维G，第三维R）对三通道依次变换
        # 200.:(69.97.143); 500.:(172,158,122); 600.:(212,133,175)
        Bool200 = (img_array[img_index] == 1)
        Bool500 = (img_array[img_index] == 2)
        Bool600 = (img_array[img_index] == 3)
        # assemble to np.dstack
        savetemp[img_index, :, :, 0] = (Bool200).astype(np.uint8) * 69 + (Bool500).astype(np.uint8) * 172 + (Bool600).astype(np.uint8) * 212
        savetemp[img_index, :, :, 1] = (Bool200).astype(np.uint8) * 97 + (Bool500).astype(np.uint8) * 158 + (Bool600).astype(np.uint8) * 133
        savetemp[img_index, :, :, 2] = (Bool200).astype(np.uint8) * 143 + (Bool500).astype(np.uint8) * 122 + (Bool600).astype(np.uint8) * 175
    # cv2.imwrite(path, savetemp)
    fig = plt.figure(figsize=figure_size, dpi=240)
    for index in range(nrows*ncols):
        ax = fig.add_subplot(nrows,ncols,index+1)
        # fig.canvas.manager.set_window_title(fig_name)
        ax.imshow(img_array[index]/ 255)
        # Attribute of sub-figure(on Fig Object)
        ax.set_title(sub_title[index], fontsize=15, fontname="Times New Roman")
        # fontname of sub-figure ticks
        x1_label = ax.get_xticklabels()
        [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
        y1_label = ax.get_yticklabels()
        [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]
        # Other attributes of sub-figure ticks
        ax.tick_params(axis='y', labelsize=8 # y轴刻度字体大小设置
                       # ,color='r',  # y轴标签颜色设置
                       # labelcolor='b',  # y轴字体颜色设置
                       # direction='in'  # y轴标签方向设置
                       )
        ax.tick_params(axis='x', labelsize=8 # y轴刻度字体大小设置
                       # ,color='r',  # y轴标签颜色设置
                       # labelcolor='b',  # y轴字体颜色设置
                       # direction='in'  # y轴标签方向设置
                       )

    # Set up seam
    # plt.subplots_adjust(wspace=0.4, hspace=0.25)
    # plt.subplots_adjust(wspace=0, hspace=0) #seamless layout
    plt.tight_layout() #tight_layout
    # Attribute of main figure(on Fig Object)
    main_title = save_filepath.split(".")[0]
    fig.suptitle(main_title, fontsize=20, fontname="Times New Roman")
    save_filepath = save_filepath.split(".")[0]+".png"
    plt.savefig(os.path.join(save_folderpath, save_filepath)) #save before show
    plt.subplots_adjust(wspace=0.4, hspace=0.25)
    # plt.show()
    # Close current figure and run next one
    # plt.pause(0.5)
    plt.close()
    print(main_title + ' is ok')

    return savetemp


def gray_normalize(image):
    """normalize image to [0, 255]"""
    img_mean = image.mean()
    img_std = image.std()
    image1 = (image-img_mean) / img_std
    img_max = image1.max()
    img_min = image1.min()
    return (image1-img_min) / (img_max-img_min) * 255.


def pngToRGBA(img_array, pred_array, nrows=1, ncols=6, figure_size=(3,15),
             save_folderpath=os.path.join(os.getcwd(),"Result", "OverlaidImage"),
             save_filepath="subject10_DE_5.png"):
    # Input: pred_array: image_num * w*h in [0, type_num-1]; img_array: w*h
    # show them with given nrows and ncols
    savetemp = np.zeros((pred_array.shape[0], pred_array.shape[1], pred_array.shape[2], 4))  # 用于存储彩色图片

    image_norm = gray_normalize(img_array)
    img_trans = np.ones_like(image_norm) * 255
    #3 channels gray enhancement images evidently
    img_png = np.stack((image_norm, image_norm, image_norm, img_trans),  axis=-1)

    for img_index in range(nrows*ncols-1): #!!!!!!!type_num
        # 布尔值矩阵（第一维B，第二维G，第三维R）对三通道依次变换
        # 200.:(69.97.143); 500.:(172,158,122); 600.:(212,133,175)
        Bool200 = (pred_array[img_index] == 1)
        Bool500 = (pred_array[img_index] == 2)
        Bool600 = (pred_array[img_index] == 3)
        Bool0 = (pred_array[img_index] != 0.)
        # assemble to np.dstack
        savetemp[img_index, :, :, 0] = (Bool200).astype(np.uint8) * 69 + (Bool500).astype(np.uint8) * 172 + (Bool600).astype(np.uint8) * 212
        savetemp[img_index, :, :, 1] = (Bool200).astype(np.uint8) * 97 + (Bool500).astype(np.uint8) * 158 + (Bool600).astype(np.uint8) * 133
        savetemp[img_index, :, :, 2] = (Bool200).astype(np.uint8) * 143 + (Bool500).astype(np.uint8) * 122 + (Bool600).astype(np.uint8) * 175
        savetemp[img_index, :, :, 3] = (Bool0).astype(np.uint8) * 210 #透明度维度
    # cv2.imwrite(path, savetemp)
    fig = plt.figure(figsize=figure_size, dpi=240)
    for index in range(nrows*ncols):
        ax = fig.add_subplot(nrows,ncols,index+1)
        # fig.canvas.manager.set_window_title(fig_name)
        # ax.imshow(img_array, cmap='gray') #first image: initial gray image
        ax.imshow(img_png/ 255.) #first image: initial gray image

        if index != 0:    #overlay RGBA label if not first image
            ax.imshow(savetemp[index-1]/ 255.)
        ax.set_xticks([]) #close ticks
        ax.set_yticks([])

    # Set up seam
    # plt.tight_layout() #tight_layout
    plt.subplots_adjust(wspace=0, hspace=0) #seamless layout
    save_filepath = save_filepath.split(".")[0]+".png"
    plt.axis('off')
    plt.savefig(os.path.join(save_folderpath, save_filepath), bbox_inches='tight') #save before show
    plt.close()

    # main_title = save_filepath.split(".")[0]
    # print(main_title + ' is ok')

    return savetemp


def write_line(foldername, filename, str, reset=False):
    #右击Jupyter输出可以直接存储为txt文件
    """write str1 to foldername/filename(line), reset at first time"""
    if not os.path.isdir(foldername): # 寻找文件foldername，如果不存在则在当前路径创建新文件
        os.makedirs(foldername)
    if reset:
        writeMode = "w"  # "w"表示覆盖读写
    else:
        writeMode = "a"  # "a"表示不覆盖续写
    with open(os.path.join(foldername, filename), writeMode) as f:
        f.write(str+"\n")


def evaluate_all_accuracy(acc_type="dice", target_tensor=None, label=None, devices="cuda"):
    if not isinstance(acc_type, list):
        acc_type = [acc_type]

    res_acc_max_tensor = torch.zeros(target_tensor.shape[0], len(acc_type), device=devices)
    res_acc_min_tensor = torch.zeros(target_tensor.shape[0], len(acc_type), device=devices)
    res_acc_avg_tensor = torch.zeros(target_tensor.shape[0], len(acc_type), device=devices)
    res_num_tensor = torch.zeros(target_tensor.shape[0], len(acc_type), device=devices)
    index = 0
    for target in target_tensor:
        accuracy_object = accuracy(target, label, acc_type, devices=devices, result_mode="average")
        res_acc_avg_tensor[index], res_num_tensor[index] = torch.tensor(accuracy_object.get_all_acc())
        accuracy_max_object = accuracy(target, label, acc_type, devices=devices, result_mode="max")
        res_acc_max_tensor[index], _ = torch.tensor(accuracy_max_object.get_all_acc())
        accuracy_min_object = accuracy(target, label, acc_type, devices=devices, result_mode="min")
        res_acc_min_tensor[index], _ = torch.tensor(accuracy_min_object.get_all_acc())
        index += 1

    return res_acc_avg_tensor, res_acc_max_tensor, res_acc_min_tensor, res_num_tensor


def evaluate_all_full_accuracy(acc_type="dice", target_tensor=None, label=None, choose_tissue=None,
                               batch_size=8, devices="cuda"):
    if not isinstance(acc_type, list):
        acc_type = [acc_type]

    res_acc_all_tensor = torch.zeros((target_tensor.shape[0], len(acc_type), batch_size),
                                     device=devices)
    index = 0
    for target in target_tensor:
        accuracy_object = accuracy(target, label, acc_type, devices=devices, is_full=True,
                                   choose_tissue=choose_tissue)
        res_acc_all_tensor[index] = accuracy_object.get_all_full_acc()
        index += 1

    return res_acc_all_tensor


def write_full_accuracy(res_acc_all_tensor, paths, acc_type="dice", txt_filename = "Accuracy_Result.txt",
                        txt_foldername = os.path.join(os.getcwd(),"Result"), is_print=False):
    """
    :param res_acc_all_tensor: Tensor(len(net_name), len(acc_type), batch_size)
    :return: write mode:   [1]path\n [2]netname:\n [3]acc_type+data
    """
    if not isinstance(acc_type, list):
        acc_type = [acc_type]

    net_num, acc_num, batch_size = res_acc_all_tensor.shape
    str_res = ""
    for batch_index in range(batch_size):
        path = paths[batch_index]
        str_res += "Image name: " + path
        if is_print:
            print(str_res)
        write_line(foldername=txt_foldername, filename=txt_filename,
                   str=str_res, reset=False)

        for net_index in range(net_num):
            str_res = "Model name: %s\n" % (net_name[net_index])
            for acc_index in range(acc_num):  # concat the result to print
                str_res += "%s: %.3f;   " % (acc_type[acc_index],
                    res_acc_all_tensor[net_index][acc_index][batch_index])

            if is_print:
                print(str_res)
            write_line(foldername=txt_foldername, filename=txt_filename,
                       str=str_res, reset=False)
        str_res = "\n\n"



#=========================================Test-set=========================================
kNum = None
batch_size = 8
txt_foldername = os.path.join(os.getcwd(),"Result")
txt_filename = "Accuracy_Result_V2.txt"

testData = dataSet(is_train=False, kNum=kNum, subType=["C0","DE","T2"])
test_iter = torch.utils.data.DataLoader(testData, batch_size, shuffle=True, drop_last=True) #由于剪裁结果一定相等，不需要指定collate_fn

str_res = ("Data loading ready! Begin to calculate accuracy!").center(100, "*")
write_line(foldername=txt_foldername, filename=txt_filename,
           str=str_res, reset=True)
print(str_res)


batch_num = len(test_iter)
num = 1
# acc_type = ["dice", "Average Symmetric Surface Diatance", "Hausdorff distance"]
acc_type = ["general dice"]

# net_name = ["UNet", "SCNN", "SRNN, L2=0.005", "SRNN", "SCSRN, 0.7SC+0.3SR", "SCSRN, 0.8SC+0.3SR"]
net_name = ["UNet_SGD", "UNet", "SCNN", "SRNN, L2=0.005", "SRNN", "SCSRN, 0.3SC+0.7SR", "SCSRN, 10C10R"]
# Average
accu_avg = torch.zeros((len(net_name), len(acc_type)), device=device)
accu_num = torch.zeros((len(net_name), len(acc_type)), device=device)
# Max and min
accu_min = 1000 * torch.ones((len(net_name), len(acc_type)), device=device)
accu_max = torch.zeros((len(net_name), len(acc_type)), device=device)
#Full
accu_full = torch.zeros((batch_num, len(net_name), len(acc_type), batch_size), device=device)


#==========For Loop: Change 4-type image-label to 2-type to calculate accuracy seperately==========
# choose_tissue = None #full tissue type
accu_full_seperate = torch.zeros((3, batch_num, len(net_name), len(acc_type), batch_size), device=device)
tissue_name = {1: "LV", 2: "RV", 3: "Myo"}


batch_index = 0 #counter of current batch_index, for accu_ful(= num - 1)
for images, labels, paths in test_iter:
    # images: torch.Size([8, 1, 240, 240]), labels: torch.Size([8, 240, 240])
    # Move data on GPU
    images = images.to(device)
    labels = labels.to(device)

    pred = net(images) #torch.Size([8, 4, 240, 240])
    pred_SGD = net_SGD(images)
    # pred_SCNN = net_SCNN(images)[0]
    pred_SRNN_L2 = net_SRNN_L2(images)
    pred_SRNN = net_SRNN(images)
    pred_SCSRN_3C7R = net_SCSRN_3C7R(images)[0]
    pred_SCSRN_10C10R = net_SCSRN_10C10R(images)[0]
    pred_SCNN_V2 = net_SCNN_V2(images)[0]
    # pred_SCSRN_5C5R = net_SCSRN_5C5R(images)[0]


#==============Calculate accuracy on test-set: torch.Size(net_num, acc_type_num)==============
    # with torch.no_grad():
    #     # target_tensor = torch.stack((pred, pred_SCNN, pred_SRNN_L2, pred_SRNN,
    #     #                             pred_SCSRN_7C3R, pred_SCSRN_8C3R), dim=0)
    #     target_tensor = torch.stack((pred_SGD, pred, pred_SCNN_V2, pred_SRNN_L2,
    #                                  pred_SRNN, pred_SCSRN_3C7R, pred_SCSRN_10C10R), dim=0)
    #     res_acc_avg_tensor, res_acc_max_tensor, res_acc_min_tensor, res_num_tensor = \
    #                 evaluate_all_accuracy(acc_type=acc_type, label=labels,
    #                                    target_tensor=target_tensor, devices=device)
    #
    # accu_avg += res_acc_avg_tensor
    # accu_max, _ = torch.max(torch.stack((res_acc_max_tensor, accu_max), dim=0), dim=0)
    # accu_min, _ = torch.min(torch.stack((res_acc_min_tensor, accu_min), dim=0), dim=0)
    # accu_num += res_num_tensor
    # print("Calculate accuracy of batch: %d/ %d successfully!" % (num, batch_num))


# ==========Return full accuracy on test-set: torch.Size(net_num, acc_type_num, batch_size)==========
# Can also calculate average, maximux and minimum use result of accu_full with file: "seperate_accuracy.npy"
# given choose_tissue:
    # None: deal with all-type at one time; Int: deal with given type seperately
    # for tissue_index in range(3):
    #     choose_tissue = tissue_index + 1
    #     txt_seperate_foldername = "Accuracy_Result_%s.txt" % (tissue_name[choose_tissue])
    #     if batch_index==0: #Reset txt-file
    #         str_firstline = ("Data loading ready! Begin to calculate %s accuracy!"
    #                     % (tissue_name[choose_tissue])).center(100, "*")
    #         write_line(foldername=txt_foldername, filename=txt_seperate_foldername,
    #                    str=str_firstline, reset=True)
    #
    #
    #     with torch.no_grad():
    #         # target_tensor = torch.stack((pred, pred_SCNN, pred_SRNN_L2, pred_SRNN,
    #         #                             pred_SCSRN_7C3R, pred_SCSRN_8C3R), dim=0)
    #         target_tensor = torch.stack((pred_SGD, pred, pred_SCNN_V2, pred_SRNN_L2,
    #                                  pred_SRNN, pred_SCSRN_3C7R, pred_SCSRN_10C10R), dim=0)
    #         res_acc_all_tensor = evaluate_all_full_accuracy(acc_type=acc_type, choose_tissue=choose_tissue,
    #                         label=labels, target_tensor=target_tensor, devices=device, batch_size=batch_size)
    #     # accu_full[batch_index] = res_acc_all_tensor
    #     # print("Calculate full mode accuracy: %d/ %d successfully!" % (num, batch_num))
    #     # batch_index += 1
    #     # write_full_accuracy(res_acc_all_tensor=res_acc_all_tensor, paths=paths, acc_type=acc_type,
    #     #                     txt_filename=txt_filename, txt_foldername=txt_foldername, is_print=False)
    #
    #     accu_full_seperate[tissue_index][batch_index] = res_acc_all_tensor
    #     print("Calculate full mode accuracy of %s: %d/ %d successfully!" %
    #                                 (tissue_name[choose_tissue], num, batch_num))
    #
    #     write_full_accuracy(res_acc_all_tensor=res_acc_all_tensor, paths=paths, acc_type=acc_type,
    #                         txt_filename=txt_seperate_foldername, txt_foldername=txt_foldername, is_print=False)
    # batch_index += 1

#=================Visualize and save image results: outputs of each network and labels=================
    # pred_lab = F.softmax(pred, dim=1).argmax(dim=1).squeeze().detach().numpy() #Array.Size([8, 1, 256, 256])
    # pred_SGD_lab = F.softmax(pred_SGD, dim=1).argmax(dim=1).squeeze().detach().numpy()  # Array
    # # pred_SCNN_lab = F.softmax(pred_SCNN, dim=1).argmax(dim=1).squeeze().detach().numpy() #Array
    # pred_SRNN_L2_lab = F.softmax(pred_SRNN_L2, dim=1).argmax(dim=1).squeeze().detach().numpy() #Array
    # pred_SRNN_lab = F.softmax(pred_SRNN, dim=1).argmax(dim=1).squeeze().detach().numpy() #Array
    # # pred_SCSRN_7C3R_lab = F.softmax(pred_SCSRN_7C3R, dim=1).argmax(dim=1).squeeze().detach().numpy() #Array
    # # pred_SCSRN_8C3R_lab = F.softmax(pred_SCSRN_8C3R, dim=1).argmax(dim=1).squeeze().detach().numpy() #Array
    # pred_SCNN_V2_lab = F.softmax(pred_SCNN_V2, dim=1).argmax(dim=1).squeeze().detach().numpy()  # Array
    # pred_SCSRN_3C7R_lab = F.softmax(pred_SCSRN_3C7R, dim=1).argmax(dim=1).squeeze().detach().numpy()  # Array
    # # pred_SCSRN_10C10R_lab = F.softmax(pred_SCSRN_10C10R, dim=1).argmax(dim=1).squeeze().detach().numpy()  # Array
    #
    # for index in range(images.shape[0]):
    #     label = labels[index]
    #     # img_array = np.stack((pred_lab[index], label.numpy().squeeze(),
    #     #            pred_SCNN_lab[index], label.numpy().squeeze(),
    #     #            pred_SRNN_L2_lab[index], label.numpy().squeeze(),
    #     #            pred_SRNN_lab[index], label.numpy().squeeze(),
    #     #            pred_SCSRN_7C3R_lab[index], label.numpy().squeeze(),
    #     #            pred_SCSRN_8C3R_lab[index], label.numpy().squeeze()), axis=0)
    #     img_array = np.stack((pred_SGD_lab[index], label.detach().numpy().squeeze(),
    #                           pred_lab[index], label.detach().numpy().squeeze(),
    #                           pred_SCNN_V2_lab[index], label.detach().numpy().squeeze(),
    #                           pred_SRNN_L2_lab[index], label.detach().numpy().squeeze(),
    #                           pred_SRNN_lab[index], label.detach().numpy().squeeze(),
    #                           pred_SCSRN_3C7R_lab[index], label.detach().numpy().squeeze()), axis=0)
    #     pngToRGB(img_array=img_array, nrows=3, ncols=4, figure_size=(9,12), save_filepath=paths[index],
    #               sub_title=["UNet_SGD", "label", "UNet", "label", "SCNN", "label",
    #                          "SRNN, L2=0.005", "label", "SRNN", "label", "SCSRN, 0.3SC+0.7SR", "label"])
    #
    # print(("Save and Show batch: %d/ %d successfully!" % (num, batch_num)).center(50, "*"))



#=================Visualize and save RGBA results: original images and outputs of each network=================
    pred_lab = F.softmax(pred, dim=1).argmax(dim=1).squeeze().detach().numpy()  # Array.Size([8, 1, 256, 256])
    pred_SGD_lab = F.softmax(pred_SGD, dim=1).argmax(dim=1).squeeze().detach().numpy()  # Array
    # pred_SCNN_lab = F.softmax(pred_SCNN, dim=1).argmax(dim=1).squeeze().detach().numpy() #Array
    pred_SRNN_L2_lab = F.softmax(pred_SRNN_L2, dim=1).argmax(dim=1).squeeze().detach().numpy()  # Array
    pred_SRNN_lab = F.softmax(pred_SRNN, dim=1).argmax(dim=1).squeeze().detach().numpy()  # Array
    # pred_SCSRN_7C3R_lab = F.softmax(pred_SCSRN_7C3R, dim=1).argmax(dim=1).squeeze().detach().numpy() #Array
    # pred_SCSRN_8C3R_lab = F.softmax(pred_SCSRN_8C3R, dim=1).argmax(dim=1).squeeze().detach().numpy() #Array
    pred_SCNN_V2_lab = F.softmax(pred_SCNN_V2, dim=1).argmax(dim=1).squeeze().detach().numpy()  # Array
    pred_SCSRN_3C7R_lab = F.softmax(pred_SCSRN_3C7R, dim=1).argmax(dim=1).squeeze().detach().numpy()  # Array
    # pred_SCSRN_10C10R_lab = F.softmax(pred_SCSRN_10C10R, dim=1).argmax(dim=1).squeeze().detach().numpy()  # Array

    for index in range(images.shape[0]):
        label = labels[index]
        img_array = images[index]

        pred_array = np.stack((label, pred_SGD_lab[index], pred_lab[index], pred_SCNN_V2_lab[index],
                              pred_SRNN_L2_lab[index], pred_SCSRN_3C7R_lab[index]), axis=0)
        pngToRGBA(img_array=img_array.squeeze(), pred_array=pred_array, nrows=1, ncols=7, figure_size=(21, 3),
                 save_filepath=paths[index])


    print(("Save and Show batch: %d/ %d successfully!" % (num, batch_num)).center(50, "*"))
    num += 1


#===================Print accuracy information of All Tissue-Types===================
# result_mode = ["average", "max", "min"]
# accu_res = torch.stack((accu_avg / accu_num, accu_max, accu_min), dim=0)
# for net_index in range(len(net_name)):
#     print("\n")
#     for acc_index in range(len(acc_type)):  # concat the result to print
#         str_res = "Model name: %s; Accuracy type: %s;\n" % (net_name[net_index], acc_type[acc_index])
#         for mode_index in range(len(result_mode)):
#             str_res += "%s: %.3f;   " % (result_mode[mode_index], accu_res[mode_index][net_index][acc_index])
#
#         print(str_res)
#         write_line(foldername=txt_foldername, filename=txt_filename,
#                    str=str_res, reset=False)
#
# np.save(os.path.join(txt_foldername, "avg_max_min_accuracy_final.npy"), accu_res.numpy())


end_time = time.time()
# print("All time of calculating accuracy function: %.3f" % (end_time_cal_acc-start_time))
print("All time: %.3f" % (end_time-start_time))


# Save Tensor Result of full accuracy:
#       Tensor(batch_num, len(net_name), len(acc_type), batch_size)
# print(accu_full)
# Save Tensor Result of full accuracy each tissue-type:
#       Tensor(3, batch_num, len(net_name), len(acc_type), batch_size)
# np.save(os.path.join(txt_foldername, "seperate_accuracy_final_V2.npy"), accu_full_seperate.numpy())

# os.system("shutdown -s -t  60 ")