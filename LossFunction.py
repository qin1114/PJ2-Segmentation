"""
@author: Menghan Qin
Input: Images: Tensor: batch_size*typenum*w*h(not sigmoid)    Labels: Tenor: batch_size*w*h - [0~typenum-1]
Output: Tensor: batch_size*1, support backward and auto grad
    Use by defining Object: CrossCompose(images_pred, labels, loss_type=Str, device) and
    call by loss_function(weight, smooth, lamda_dice, lamda_cross),
    You can sum the output up to avoid sum in train function, but sum in train function can deal with the different
    size caused the last batch.
"""

import torch
from torch import nn
from torch.nn import functional as F
from core.Net import SRNN


class CrossCompose(object):
    """
    定义所有需要使用的损失函数，传入一个batch的输入
    images: batch_size*typenum*w*h;     labels: batch_size*w*h
    """
    def __init__(self, images_pred, labels, loss_type="lossCrossEntropy", regular_type=None,
                 device="cuda", SR_path=None, use_SR=False, pos_out=None, gold_pos=None):
        self.size = images_pred.shape
        if len(self.size )==3: #batch_size为1导致维度降低
            self.images_pred = images_pred.unsqueeze(0)
            self.labels = labels.unsqueeze(0)
            self.size = images_pred.shape
        else:
            self.images_pred = images_pred.to(device)
            self.labels = labels.to(device)

        self.batch_size = self.size[0]
        self.type_num = self.size[1]
        self.loss_type = loss_type
        self.regular_type = regular_type
        self.device = device
        self.SR_path = SR_path
        self.use_SR = use_SR
        self.pos_out = pos_out
        self.gold_pos = gold_pos


    def weight_map(self, weight_type=None):
        """calculate weight matrix of WeightedCrossEntropy and GeneralizedDiceLoss
        images in same batch share the weights"""
        if weight_type == "lossWeightedCrossEntropy":
            """calculate on y_preds"""
            y_preds = torch.zeros(self.size).to(self.device)
            for item in range(self.batch_size):  # 在类别维应用激活函数，获取属于各类的概率
                # detach: 权重的所有计算不需要涉及梯度
                y_preds[item, :, :, :] = F.softmax(self.images_pred[item, :, :, :].detach(), dim=0)
            N = self.size[0] * self.size[2] * self.size[3]
            y_sum = torch.sum(y_preds, (0,2,3))
            # m = (N-y_sum)/y_sum/torch.sum((N-y_sum)/y_sum) #归一化
            m = (N - y_sum) / y_sum

        elif weight_type == "multiGeneralizedDiceLoss":
            """calculate on labels"""
            labels = torch.zeros(self.size, requires_grad=False).to(self.device)
            for type_index in range(self.type_num):
                labels[:, type_index, :, :][self.labels.detach() == type_index] = 1
            y_sum = torch.sum(labels, (0,2,3))
            m = (1/ (y_sum*y_sum)/ torch.sum(1/ (y_sum*y_sum))).unsqueeze(1) #Normalize to ensure precision
            # m = (1 / (y_sum * y_sum)).unsqueeze(1)

        return m


    def lossCrossEntropy(self):
        # 只能处理targets为int64类型，返回结果为对数交叉熵，将softmax-log-NLLLoss合并到一块
        # print(F.cross_entropy(self.images_pred, torch.Tensor.long(self.labels), reduction='none'))
        return F.cross_entropy(self.images_pred, torch.Tensor.long(self.labels), reduction='none').mean(1).mean(1)


    def lossWeightedCrossEntropy(self, weight):
        # 返回结果为加权对数交叉熵
        if weight == None:
            weight = self.weight_map(weight_type="lossWeightedCrossEntropy")
        return F.cross_entropy(self.images_pred, torch.Tensor.long(self.labels), weight, reduction='none').mean(1).mean(1)


    def multiDiceLoss(self, smooth, weight=None):
        if self.loss_type in ["multiGeneralizedDiceLoss", "weighted_cross_entropy+generalized_dice_loss"]:
            if weight == None:
                weight = self.weight_map(weight_type="multiGeneralizedDiceLoss")
        elif self.loss_type not in ["multiDiceLoss", "CrossEntropy+Dice"]:
            # weight = (torch.ones(self.type_num) / self.type_num).unsqueeze(1).to(self.device)
            print("Illegal loss type input, please check {} again!".format(self.loss_type))

        labels = torch.zeros(self.size, requires_grad=False).to(self.device) #创建空矩阵，不计算梯度，用于存储标签的32*4*w*h结果
        for type_index in range(self.type_num):  # 一般==在判断时即使不同类型数据也可以为true，主要看数值
            labels[:, type_index, :, :][self.labels == type_index] = 1 # 扩展label为typenum维
        # 在类别维应用激活函数获取分类结果，softmax层默认第一维是batch_size
        y_preds = F.softmax(self.images_pred, dim=1).to(torch.float32).to(self.device)
        # print(torch.sum(result[1], axis=0)) #验证结果
        # y_preds = self.images_pred.to(torch.float32).to(self.device)

        labels = torch.flatten(labels, start_dim=2) #重整结果，将尺寸维压缩
        y_preds = torch.flatten(y_preds, start_dim=2) #重整结果，将尺寸维压缩

        intersection = torch.sum(y_preds*labels, axis=2) #先求交集，对应项点乘再相加，得32*4
        unionset = torch.sum(y_preds*y_preds, axis=2) + \
                   torch.sum(labels*labels, axis=2)# 求并集，直接分别平方后求和，再将两个输入进行相加
        if self.loss_type in ["multiGeneralizedDiceLoss", "weighted_cross_entropy+generalized_dice_loss"]:
            sum_dice_loss = 1-2*torch.mm(intersection, weight).squeeze()/ \
                            (torch.mm(unionset, weight).squeeze()+smooth)  #torch.mm应用矩阵乘法将权重加权
        else: #不施加权重
            sum_dice_loss = 1 - torch.sum(2*intersection/ (unionset+smooth), dim=1) / self.type_num
        return sum_dice_loss #32*1


    def loss_term(self, weight = None, smooth=1e-5, lamda_dice=0.7, lamda_cross=0.3):
        """Input: Images: Tensor: batch_size*typenum*w*h(not sigmoid)
             Labels: Tenor: batch_size*w*h - [0~typenum-1]
          Output: Tensor: batch_size, will be sum up and divided in train_function"""
        if self.loss_type=="lossCrossEntropy":
            return self.lossCrossEntropy()

        elif self.loss_type=="lossWeightedCrossEntropy":
            return self.lossWeightedCrossEntropy(weight)

        elif self.loss_type in ["multiDiceLoss", "multiGeneralizedDiceLoss"]:
            return self.multiDiceLoss(smooth, weight)

        elif self.loss_type=="CrossEntropy+Dice":
            return self.lossCrossEntropy()*lamda_cross + self.multiDiceLoss(smooth, weight)*lamda_dice

        elif self.loss_type=="weighted_cross_entropy+generalized_dice_loss":
            return self.lossWeightedCrossEntropy(weight)*lamda_cross + \
                self.multiDiceLoss(smooth, weight).squeeze()*lamda_dice

        else:
            print("Illegal loss type input, please check {} again!".format(self.loss_type))


    def labels_to_onehot(self, target):
        """change target from: Int(0,1,2,3) to: Typenum(binary)"""
        shape = target.shape
        onehot_tensor = torch.zeros((self.batch_size, self.type_num, shape[1], shape[2])).to(self.device)
        for type_index in range(self.type_num):
            onehot_tensor[:, type_index] = (target==type_index).to(torch.float32)
        return onehot_tensor


    def SR_regular_term(self):
        """Compute SR regular term"""
        SRNet = SRNN().to(self.device)
        SRNet.load_state_dict(torch.load(self.SR_path))
        """change Unet Torch output: Float(batch_size*type_num*w*h),
        to Float(batch_size*type_num*w*h) [0~1], 
        then to Int(batch_size*1*w*h) [0,1,2,3]"""
        y_preds = F.softmax(self.images_pred, dim=1).argmax(1, keepdim=False) \
            .to(torch.float32).to(self.device)
        y_onehot = self.labels_to_onehot(y_preds)
        R_hat = SRNet(y_onehot).reshape(self.batch_size, -1).to(torch.float32)  # batch_size*(type_num*w*h)
        label_onehot = self.labels_to_onehot(self.labels.detach().squeeze().to(torch.float32))
        R = SRNet(label_onehot).reshape(self.batch_size, -1)
        res_tensor = torch.norm((R_hat - R), dim=1)  # L2 Regular: the same shape
        return res_tensor * res_tensor


    def SC_regular_term(self):
        """Compute SC regular term"""
        P_hat = self.pos_out  # batch_size*1
        P = self.gold_pos.to(torch.float32).reshape(-1, 1)
        res_tensor = (P_hat - P) * (P_hat - P)
        return res_tensor


    def regular_term(self, lamda_SR=5e-4, lamda_SC=1e-6):
        if self.regular_type == None:
            return torch.zeros(self.batch_size).to(self.device)

        elif self.regular_type == "SR":
            return self.SR_regular_term() * lamda_SR

        elif self.regular_type == "SC":
            return self.SC_regular_term() * lamda_SC

        elif self.regular_type in ["SC+SR", "SR+SC"]:
            weight_SR = 0.5
            weight_SC = 0.5
            return lamda_SR*weight_SR * self.SR_regular_term() + lamda_SC*weight_SC * self.SC_regular_term()
        else:
            print("Illegal regular type input, please check {} again!".format(self.regular_type))


    def loss_function(self):
        loss_part = self.loss_term()
        regular_part = self.regular_term()
        percentage = loss_part.sum() / (1e-6+regular_part).sum()
        return loss_part + regular_part, percentage



"""test loss"""
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
# gt = torch.Tensor([
#     [[2, 0, 0, 2],
#      [0, 2, 2, 0],
#      [0, 1, 1, 0],
#      [2, 0, 0, 2]],
#     [[2, 0, 0, 2],
#      [0, 2, 2, 0],
#      [0, 1, 1, 0],
#      [2, 0, 0, 2]]
# ])
#
#
# # print(pred.shape, gt.shape)
# multi_dice = loss_fun(pred1, gt, loss_type="multiGeneralizedDiceLoss", device=try_all_gpus()[0])
# print(multi_dice.loss_function()) #<class 'numpy.float64'>