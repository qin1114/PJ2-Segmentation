"""
@author: Menghan Qin
Input: Images: Tensor: batch_size*typenum*w*h(not sigmoid)    Labels: Tenor: batch_size*w*h - [0~typenum-1]
    Use by defining Object: AccuracyCompose(images, targets, acc_type) and
    call by get_all_acc(smooth), You can modify the trans_targets Function to deal with different input-type
    given result_mode if you want to return other mode: AccuracyCompose(images, targets, acc_type, result_mode)

    Use by defining Object: AccuracyCompose(images, targets, acc_type, is_full) and
    call by get_all_full_acc(smooth) if you want to get full accuracy result and then visualize.


PS: If you want to calculate accuracy of each tissue-type seperately, I recommend you changing your image-label
    in "trans_target_seperate" Method according to introduction above, due to complexion of change accuracy function.
"""

from d2l import torch as d2l
import torch
import sklearn.metrics
import scipy.ndimage
import numpy as np
from torch.nn import functional as F


np.set_printoptions(threshold=np.inf)

class AccuracyCompose(object):
    """Input: Images: Tensor: batch_size*typenum*w*h    Labels: Tenor: batch_size*w*h - [0~typenum-1]
    Change input type: softmax on images; labels to binary: batch_size*typenum*w*h
    Use by defining Object: AccuracyCompose(images, targets, acc_type) and
    call by get_all_acc(smooth), You can modify the trans_targets Function to deal with different input-type"""
    def __init__(self, images, targets, acc_type, choose_tissue=None,
                 result_mode="average", is_full=False, devices="cpu"):
        self.acc_type = acc_type #list of required acc
        self.type_num = images.shape[1]
        self.images = F.softmax(images, dim=1).to(devices) #return batch_size*typenum*w*h - [0, 1]
        # self.images = images.to(devices) #For test Acc-function
        self.devices = devices
        self.choose_tissue = choose_tissue  # choose type of tissue to calculate accuracy
        if choose_tissue==None:
            self.targets = self.trans_targets(targets) #return batch_size*typenum*w*h - [0, 1]
        else:
            self.targets = self.trans_targets_seperate(targets)  # return batch_size*2*w*h - [0, 1]
            self.images = self.trans_images_seperate(self.images) # return batch_size*2*w*h - [0~1]
            self.type_num = 2

        self.batch_size = images.shape[0]
        self.result_mode = result_mode #return average or max or min
        self.is_full = is_full #return tensor of batch_size


    def trans_targets(self, targets):
        """transfer labels from Tenor: batch_size*w*h - [0~typenum-1] to
        Tensor: batch_size*typenum*w*h - binary"""
        target_binary = torch.zeros(self.images.shape, device=self.devices)
        for type_index in range(self.type_num):
            target_binary[:, type_index] = (targets[:]==type_index).int() #Bool to int
        return target_binary


    def trans_images_seperate(self, images):
        """transfer images from Tensor: batch_size*typenum*w*h - Float[0~1] to
                Tensor: batch_size*2*w*h - Float[0~1], for calculate accuracy for chosen-tissue"""
        return torch.stack((images[:, self.choose_tissue],
                            1-images[:, self.choose_tissue]), dim=1)


    def trans_targets_seperate(self, targets):
        """transfer images from Tensor: Tenor: batch_size*w*h - [0~typenum-1] to
                        Tensor: batch_size*2*w*h - binary, for calculate accuracy for chosen-tissue"""
        target_binary = (targets==self.choose_tissue).int()
        return torch.stack((target_binary, 1 - target_binary), dim=1) #1 to 0 and 0 to 1(binary)


    def get_acc(self, acc_type, smooth=1e-5):
        if acc_type == "Hausdorff distance":
            return Surface(self.images, self.targets, self.devices, result_mode=self.result_mode
                           , is_full=self.is_full).Hausdorff_dis()
        elif acc_type == "Average Symmetric Surface Diatance":
            return Surface(self.images, self.targets, self.devices, result_mode=self.result_mode
                           , is_full=self.is_full).ASSD()
        elif acc_type == "correct prediction of pixel":
            return accuracy_pixel(self.images, self.targets, device=self.devices, result_mode=self.result_mode)
        elif acc_type == "dice":
            return dice(self.images, self.targets, smooth=smooth, result_mode=self.result_mode
                        , is_full=self.is_full, device=self.devices)
        elif acc_type == "general dice":
            return general_dice(self.images, self.targets, smooth=smooth, result_mode=self.result_mode
                        , is_full=self.is_full, device=self.devices)
        elif acc_type == "specificity":
            return AUC_Accuracy(self.images, self.targets, self.devices, result_mode=self.result_mode
                                ).speci(smooth=smooth)
        elif acc_type == "sensitivity":
            return AUC_Accuracy(self.images, self.targets, self.devices, result_mode=self.result_mode
                                ).sensi(smooth=smooth)
        elif acc_type == "precision":
            return AUC_Accuracy(self.images, self.targets, self.devices, result_mode=self.result_mode
                                ).precision(smooth=smooth)
        elif acc_type == "accuracy":
            return AUC_Accuracy(self.images, self.targets, self.devices, result_mode=self.result_mode
                                ).accuracy(smooth=smooth)

        else:
            print("Illegal accuracy type input:{}, please check it!".format(acc_type))


    def get_all_acc(self, smooth=1e-5):
        """traverse the acc_type str-list, and return list of required acc
        Used mainly for train and valid"""
        acc_list, num_list = [], []
        for type_item in self.acc_type:
            if type_item not in ["Hausdorff distance", "Average Symmetric Surface Diatance"]:
                acc_list.append(self.get_acc(type_item, smooth))  # array to float
                num_list.append(self.batch_size)
            else:
                acc_list.append(self.get_acc(type_item, smooth)[0])
                num_list.append(self.get_acc(type_item, smooth)[1])
        return acc_list, num_list


    def get_all_full_acc(self, smooth=1e-5):
        """
        Used mainly for visualization on test set:
            the same as get_all_acc, but return tensor of batch_size(of required acc)
        Input: return of self.get_acc each time: Tensor(batch_size)
        Output: Tensor(len(acc_type), batch_size)
        PS: Only support ["dice", "Hausdorff distance", "Average Symmetric Surface Diatance"],
            Need to expand return value of other accuracy function to support more!
        """
        acc_tensor = torch.zeros((len(self.acc_type), self.batch_size), device=self.devices)
        type_index = 0
        for type_item in self.acc_type:
            if type_item in ["dice", "Hausdorff distance", "Average Symmetric Surface Diatance",
                             "general dice"]:
                acc_tensor[type_index] = self.get_acc(type_item, smooth)
            else:
                print("Illegal full mode accuracy type input:{}, please check it!".format(type_item))
            type_index += 1

        return acc_tensor



"""
Definition of all Accuracy Functions are listed below.
Accuracy Matrix: based on images and labels with Tenser: [batch_size*typenum*w*h]~[0，1]
                output: sum(batch_size*1)
True Positive （真正， TP）预测为正的正样本
True Negative（真负 , TN）预测为负的负样本
False Positive （假正， FP）预测为正的负样本
False Negative（假负 , FN）预测为负的正样本
"""

class AUC_Accuracy(object):
    def __init__(self, images, labels, device="cuda", result_mode="average"):
        self.labels = labels
        self.shape = images.shape
        self.device = device
        self.images = self.trans_targets(images)
        self.result_mode = result_mode

    def trans_targets(self, targets):
        """transfer images from Tenor: batch_size*typenum*w*h - float[0~1] to
        Tensor: batch_size*typenum*w*h - binary"""
        type_num = self.shape[1]
        target_binary = torch.zeros(self.shape, device=self.device)
        images_pred = targets.argmax(axis=1)
        for type_index in range(type_num):
            target_binary[:, type_index] = (images_pred[:] == type_index).int()  # Bool to int
        return target_binary

    def get_classify(self, images, labels):
        """return Tensor: batch_size"""
        true_matrix = (labels[:, 0] == 0)  # reverse of type0, batch_size*w*h
        postive_matrix = (images[:, 0] == 0)
        TP = torch.sum(true_matrix & postive_matrix, dim=(1, 2))
        TN = torch.sum(true_matrix & ~postive_matrix, dim=(1, 2))
        FP = torch.sum(~true_matrix & postive_matrix, dim=(1, 2))
        FN = torch.sum(~true_matrix & ~postive_matrix, dim=(1, 2))
        return (TP, TN, FP, FN)

    def speci(self, smooth=1e-5):
        """calculate specificity: TN/(TN+FP); return constant: sum"""
        _, TN, FP, _ = self.get_classify(self.images, self.labels)
        if self.result_mode=="average":
            return torch.sum(TN / (TN + FP + smooth))
        # elif self.result_mode=="max":
        #     return torch.max(TN / (TN + FP + smooth))[0]
        # elif self.result_mode == "min":
        #     return torch.min(TN / (TN + FP + smooth))[0]
        # else:
        #     print("Illegal result mode input:{}, please check it!".format(self.result_mode))

    def sensi(self, smooth=1e-5):
        """calculate sensitivity(also called recall): TP/(TP+FN); return constant: sum"""
        TP, _, _, FN = self.get_classify(self.images, self.labels)
        return torch.sum(TP / (TP + FN + smooth))


    def precision(self, smooth=1e-5):
        """calculate precision: TP/(TP+FP); return constant: sum"""
        TP, _, FP, _ = self.get_classify(self.images, self.labels)
        return torch.sum(TP / (TP + FP + smooth))


    def accuracy(self, smooth=1e-5):
        """calculate accuracy: (TP+TN)/(TP+FP+FN+TN); return constant: sum"""
        TP, TN, FP, FN = self.get_classify(self.images, self.labels)
        return torch.sum((TP + FN) / (TP + FP + FN + TN + smooth))


    def cal_auc(self):
        """calculate auc; return constant: sum"""
        auc = 0
        for item in range(self.images.shape[0]):
            images_temp = torch.flatten(self.images[item], start_dim=1).transpose(1, 0)
            labels_temp = torch.flatten(self.labels[item], start_dim=1).transpose(1, 0)
            auc += sklearn.metrics.roc_auc_score(labels_temp, images_temp)
        return auc



def dice(images, labels, smooth=1e-5, result_mode="average", is_full=False, device="cuda"):
    """calculate dice of multi-type prediction result"""
    batch_size, type_num = images.shape[0:2]
    sum_dice, max_dice, min_dice = 0, 0, 10000
    full_dice = torch.zeros(batch_size, device=device)
    for batch_index in range(batch_size):
        temp_index = 0 #record dice of each batch_image
        for type_index in range(type_num):
            intersection = torch.sum(images[batch_index, type_index] * labels[batch_index, type_index])
            unionset = torch.sum(images[batch_index, type_index] * images[batch_index, type_index]) + \
                       torch.sum(labels[batch_index, type_index] * labels[batch_index, type_index])
            temp_index += 2 * intersection / (unionset + smooth) / type_num
        sum_dice += temp_index
        max_dice = max((temp_index, max_dice))
        min_dice = min((temp_index, min_dice))
        if is_full:
            full_dice[batch_index] = temp_index

    if not is_full:
        if result_mode == "average":
            return sum_dice
        elif result_mode == "max":
            return max_dice
        elif result_mode == "min":
            return min_dice
        else:
            print("Illegal result mode input:{}, please check it!".format(result_mode))
    else:
        return full_dice


def general_dice(images, labels, smooth=1e-5, result_mode="average", is_full=False, device="cuda"):
    """calculate general dice of multi-type prediction result"""
    batch_size, type_num = images.shape[0:2]

    y_sum = torch.sum(labels, (0, 2, 3))
    weight = (1 / (y_sum * y_sum) / torch.sum(1 / (y_sum * y_sum))).unsqueeze(1)  # Normalize to ensure precision

    labels = torch.flatten(labels, start_dim=2)  # 重整结果，将尺寸维压缩
    y_preds = torch.flatten(images, start_dim=2)  # 重整结果，将尺寸维压缩
    intersection = torch.sum(y_preds * labels, axis=2)  # 先求交集，对应项点乘再相加，得32*4
    unionset = torch.sum(y_preds * y_preds, axis=2) + \
               torch.sum(labels * labels, axis=2)  # 求并集，直接分别平方后求和，再将两个输入进行相加
    full_general_dice = 1 - 2 * torch.mm(intersection, weight).squeeze() / \
                        (torch.mm(unionset, weight).squeeze() + smooth)  # torch.mm应用矩阵乘法将权重加权

    if not is_full:
        if result_mode == "average":
            return full_general_dice.mean()
        elif result_mode == "max":
            return full_general_dice.max()
        elif result_mode == "min":
            return full_general_dice.min()
        elif result_mode == "std":
            return full_general_dice.std()
        else:
            print("Illegal result mode input:{}, please check it!".format(result_mode))
    else:
        return full_general_dice


def accuracy_pixel(images, labels, device="cpu", result_mode="average"):
    """calculate the accuracy average of each pixel(of each batch) for multi-type,
    return: Constant sum of batch"""
    type_num = images.shape[1]
    num_pixel = images.shape[2] * images.shape[3]
    images = images.argmax(axis=1) #transfer to binary
    acc_sum = torch.zeros(images.shape[0]).to(device)
    for i in range(type_num):
        temp = (images==i) + labels[:,i] #both to be 1
        acc_sum += (torch.sum(temp==2, dim=(1,2))/ num_pixel).float()

    if result_mode == "average": #sum up batch
        return torch.sum(acc_sum)
    else:
        print("Illegal result mode input:{}, please check it!".format(result_mode))


class Surface(object):
    """Only take 1 exits into consideration, and pass labels(pred or gold) to be all 0"""
    def __init__(self, images, labels, device="cuda", result_mode="average", is_full=False):
        self.shape = images.shape
        self.device = device
        self.type_num = self.shape[1]
        self.images = self.trans_targets(images)
        self.labels = labels
        self.batch_size = self.shape[0]
        self.result_mode = result_mode
        self.is_full = is_full

    def trans_targets(self, targets):
        """transfer images from Tenor: batch_size*typenum*w*h - float[0~1] to
        Tensor: batch_size*typenum*w*h - binary"""
        target_binary = torch.zeros(self.shape, device=self.device)
        images_pred = targets.argmax(axis=1)
        for type_index in range(self.type_num):
            target_binary[:, type_index] = (images_pred[:] == type_index).int()  # Bool to int
        return target_binary

    def get_contour(self, target):
        """Input: Tensor: type_num*w*h: [0, 1]"""
        # target_tensor = torch.sum(target[:,1:], dim=1) #return batch_size*w*h
        target_tensor = 1-target[0] #return w*h
        target_array = np.uint8(target_tensor.cpu().numpy()) #numpy only support CPU
        erosed_bool = scipy.ndimage.binary_erosion(target_array, iterations=1).astype(target_array.dtype)
        contour_array = target_array - erosed_bool #concour array with the value of 0 or 1
        row_index, col_index = np.where(contour_array == 1)#array of coordinate respectively

        return np.dstack((row_index, col_index))[0] #return len(contour)*2

    def get_contour_len(self, target):
        return len(self.get_contour(target))

    def get_mindis_A_to_B(self, A, B):
        """
        Input: Tensor: typenum*w*h  Output: Array: (w*h)*1
        return the min distance between points in B and a given A point
        """
        contourA_index = self.get_contour(A) #tuple of row_index and col_index: num_edge-points * 2
        contourB_index = self.get_contour(B)  # tuple of row_index and col_index

        mytree = scipy.spatial.cKDTree(contourA_index)
        dist_ckd, _ = mytree.query(contourB_index, p=2) # calculate minimum Euclid distance
        return dist_ckd #Array: (w*h)*1

    def Hausdorff_dis(self):
        """calculate Hausdorff distance"""
        hau_dis, contour_len = 0, 0
        hau_max, hau_min = 0, 100
        hau_full = torch.zeros(self.batch_size, device=self.device)

        for item in range(self.batch_size):
            temp_dis = 0
            if self.get_contour_len(self.images[item])!=0 and self.get_contour_len(self.labels[item])!=0:
                maxdis_image_to_labels = np.max(self.get_mindis_A_to_B(self.images[item], self.labels[item]))
                maxdis_label_to_images = np.max(self.get_mindis_A_to_B(self.labels[item], self.images[item]))
                # print(2, maxdis_image_to_labels, maxdis_label_to_images)
                temp_dis += np.maximum(maxdis_image_to_labels, maxdis_label_to_images) #accumulate of batch_size
                hau_dis += temp_dis
                hau_max = max(hau_max, temp_dis)
                hau_min = min(hau_min, temp_dis)
                if self.is_full:
                    hau_full[item] = temp_dis
                contour_len += 1
            elif self.is_full: #Tag it to be -1, to ensure length to be batch_size
                hau_full[item] = -1

        if not self.is_full:
            if self.result_mode == "average":
                return hau_dis, contour_len
            elif self.result_mode == "max":
                return hau_max, contour_len
            elif self.result_mode == "min":
                return hau_min, contour_len
            else:
                print("Illegal result mode input:{}, please check it!".format(self.result_mode))
        else:
            return hau_full


    def ASSD(self):
        """calculate Average Symmetric Surface Distance"""
        assd_dis, contour_len = 0, 0
        adds_max, adds_min = 0, 100
        adds_full = torch.zeros(self.batch_size, device=self.device)

        for item in range(self.batch_size):
            temp_dis = 0
            image_contour_len = self.get_contour_len(self.images[item])
            label_contour_len = self.get_contour_len(self.labels[item])
            if image_contour_len!=0 and label_contour_len!=0:
                sumdis_image_to_labels = np.sum(self.get_mindis_A_to_B(self.images[item], self.labels[item]))
                sumdis_label_to_images = np.sum(self.get_mindis_A_to_B(self.labels[item], self.images[item]))
                # print(1, sumdis_image_to_labels, sumdis_label_to_images)
                temp_dis += (sumdis_image_to_labels + sumdis_label_to_images) / (image_contour_len+label_contour_len)
                assd_dis += temp_dis
                adds_max = max(adds_max, temp_dis)
                adds_min = min(adds_min, temp_dis)
                contour_len += 1
                if self.is_full:
                    adds_full[item] = temp_dis
            elif self.is_full: #Tag it to be -1, to ensure length to be batch_size
                adds_full[item] = -1

        if not self.is_full:
            if self.result_mode == "average":
                return assd_dis, contour_len
            elif self.result_mode == "max":
                return adds_max, contour_len
            elif self.result_mode == "min":
                return adds_min, contour_len
            else:
                print("Illegal result mode input:{}, please check it!".format(self.result_mode))
        else:
            return adds_full


# pred = torch.Tensor([[
#         [[0, 1, 0, 0],
#          [1, 0, 0, 1],
#          [1, 0, 0, 1],
#          [0, 1, 1, 0]],
#         [[0, 0, 0, 0],
#          [0, 0, 0, 0],
#          [0, 1, 1, 0],
#          [0, 0, 0, 0]],
#         [[1, 0, 1, 1],
#          [0, 1, 1, 0],
#          [0, 0, 0, 0],
#          [1, 0, 0, 1]]],
#         [
#             [[0, 1, 0, 0],
#              [1, 0, 0, 1],
#              [1, 0, 0, 1],
#              [0, 1, 1, 0]],
#             [[0, 0, 0, 0],
#              [0, 0, 0, 0],
#              [0, 1, 1, 0],
#              [0, 0, 0, 0]],
#             [[1, 0, 1, 1],
#              [0, 1, 1, 0],
#              [0, 0, 0, 0],
#              [1, 0, 0, 1]]]
#     ])
#
# pred1 = torch.Tensor([[
#     [[0, 1, 1, 0],
#      [1, 0, 0, 1],
#      [1, 0, 0, 1],
#      [0, 1, 1, 0]],
#     [[0, 0, 0, 0],
#      [0, 0, 0, 0],
#      [0, 1, 1, 0],
#      [0, 0, 0, 0]],
#     [[1, 0, 0, 1],
#      [0, 1, 1, 0],
#      [0, 0, 0, 0],
#      [1, 0, 0, 1]]],
#     [
#         [[0, 1, 1, 0],
#          [1, 0, 0, 1],
#          [1, 0, 0, 1],
#          [0, 1, 1, 0]],
#         [[0, 0, 0, 0],
#          [0, 0, 0, 0],
#          [0, 1, 1, 0],
#          [0, 0, 0, 0]],
#         [[1, 0, 0, 1],
#          [0, 1, 1, 0],
#          [0, 0, 0, 0],
#          [1, 0, 0, 1]]]
# ])
#
#
# gt = torch.Tensor([[
#     [[0, 1, 1, 0],
#      [1, 0, 0, 1],
#      [1, 0, 0, 1],
#      [0, 1, 1, 0]],
#     [[0, 0, 0, 0],
#      [0, 0, 0, 0],
#      [0, 1, 1, 0],
#      [0, 0, 0, 0]],
#     [[1, 0, 0, 1],
#      [0, 1, 1, 0],
#      [0, 0, 0, 0],
#      [1, 0, 0, 1]]],
#     [
#         [[0, 1, 1, 0],
#          [1, 0, 0, 1],
#          [1, 0, 0, 1],
#          [0, 1, 1, 0]],
#         [[0, 0, 0, 0],
#          [0, 0, 0, 0],
#          [0, 1, 1, 0],
#          [0, 0, 0, 0]],
#         [[1, 0, 0, 1],
#          [0, 1, 1, 0],
#          [0, 0, 0, 0],
#          [1, 0, 0, 1]]]
# ])
#
# # print(pred.shape, gt.shape)
# mySurface = Surface(pred, gt) #pred1 is the same as gt, while pred is a little different
# print(type(mySurface.ASSD())) #<class 'numpy.float64'>




