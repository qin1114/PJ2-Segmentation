"""
@author: Menghan Qin
图像变换主要两个功能：数据集过小时通过随即变换增加图片量；图片增强在每个epoch中引入随机性
考虑随机剪裁并resize为标准大小、随机旋转、随机翻转、归一化和缩小内存，采用以下三种方式
1、通过随机剪裁和resize以及归一化并缩小内存获得增加后的数据，应用随机翻转和随机旋转在图片增强中
2、直接在图片增广时应用四种变化(不能增大数据量)，使用新定义的类singCropResize(仿照原有的ransform的RandomResizedCrop函数)
3、全部用于增加图片量并存储到本地，此时考虑到样本大小随机需要重新定义collate_fn：内存占用过大
最后选用模式2，通过NPYSegDataset(is_train)和torch.utils.data.DataLoader(trainData, batch_size, shuffle=True,
drop_last=True, collate_fn=collate_fn)调用
"""
import torchvision
import os
import torch
import numpy as np
import cv2
from torchvision import transforms
from torchvision.transforms import functional as tF
from PIL import Image
import random
import time
import math

c0_lge_t2_mean_std = {"C0":(398.816, 395.903), "DE":(242.600, 158.449), "T2":(164.044, 182.646)} #均值和标准差
labelMap = {200.: 1, 500.:2, 600.:3}

class randCropResize(object):
    '''
    随机切割：先按照尺寸缩减比例获取位置随机参数，再应用参数，要求必须为PIL格式
    需要注意的是resize过程并不包含随机性，所以可以直接调用函数
    '''
    def __init__(self, sizeList, crop_size, is_expand=True):  # 在init中接受外来参数，设置为属性，并且在call方法中调用。
        self.sizeList = sizeList
        self.crop_size = crop_size
        self.is_expand = is_expand #由于裁剪的随机性存在于位置参数中，必须要expand，不能重复裁剪

    def __call__(self, image, label):
        if not isinstance(image, list):  # 传入的image是一个以array为元素的列表
            image, label = [image], [label]  # 说明输入单张图片(不能对列表在循环遍历时append，故只能遍历原始图集)
        # imageList, labelList = image, label #需要保留原始图片集，
        imageList, labelList = [], []  # 由于原始图片也要resize，我们将1也作为sizeList传入并参与下面的循环
        num = len(image)

        size = image[0].shape  # 图片尺寸
        for index in range(num):  # 遍历每一张图片
            imageCur = Image.fromarray(image[index])  # ndarray转为PIL
            labelCur = Image.fromarray(label[index])
            for item in self.sizeList:
                curSize = (int(size[0] * item), int(size[1] * item))
                crop_params = transforms.RandomCrop.get_params(imageCur, curSize)  # 获取随机的位置参数
                imageTemp = tF.crop(imageCur, *crop_params)  # 为了不覆盖imageCur
                labelTemp = tF.crop(labelCur, *crop_params)
                # print(imageTemp.mode) #"F":32浮点型图片
                imageTemp = np.asarray(imageTemp)  # 重新转化回ndarray
                labelTemp = np.asarray(labelTemp)
                # 记录返回值列表(由于compose不改变传入传出格式，注意全部调为ndarray)
                if (imageTemp.shape[0] >= self.crop_size[0]) and (imageTemp.shape[1] >= self.crop_size[1]):
                    imageList.append(cv2.resize(imageTemp, self.crop_size))  # 对crop结果进行resize(验证发现不会改变label的值)
                    labelList.append(cv2.resize(labelTemp, self.crop_size))
                    # print(cv2.resize(imageTemp, self.crop_size).shape)
                else:
                    pass  # 过滤掉不够大的图片，插值会产生无效的label像素点
                # 只可以创建文件，需要确保文件夹存在
                # cv2.imwrite(os.path.join(r".\temp\Cropshort", "crop"+str(item)+"_img"+str(index) + "_image.png"), imageTemp)
                # cv2.imwrite(os.path.join(r".\temp\Cropshort", "crop"+str(item)+"_img"+str(index) + "_label.png"), labelTemp)
        # print(len(imageList))
        return imageList, labelList  # compose要求有返回值，没有返回值将会报错


class randFlip(object):
    """
    进行随机翻转
    is_expand决定了模式，如果为True则对每一张图片分别按照不同的方式进行随机翻转(每张图片只操作一种变换)，将结果存储，列表进行传递
    如果为False则在每个epoch中引入随机性，每次只传入一张图片，对该图片依次应用各种随机方法并依旧返回该图片(注意减小变化的概率)
    """

    def __init__(self, flipCodes, is_expand=True):
        self.flipCodes = flipCodes
        self.is_expand = is_expand  # True则使用扩展图片源，False则在epoch中引入图片增广

    def __call__(self, image, label):
        if not isinstance(image, list):  # 传入的image是一个以array为元素的列表
            image, label = [image], [label]
        imageList, labelList = image, label  # 需要保留原始图片集
        num = len(image)

        for index in range(num):  # 遍历每一张图片，如果打开is_expand模式则此时默认一个
            imageTemp, labelTemp = image[index], label[index]
            for item in self.flipCodes:
                if random.random() > 0.7:  # 预先判断是否进行翻转
                    imageTemp = cv2.flip(imageTemp, item)
                    labelTemp = cv2.flip(labelTemp, item)
                    if self.is_expand:
                        imageList.append(imageTemp)
                        labelList.append(labelTemp)
                        imageTemp, labelTemp = image[index], label[index]  # 保证变化之间相互独立，即各种反转变化只进行一次
                    else:# 对原始图片连续变化(在index基础上变化则可以处理列表的累计变化，相当于修改原始图集)
                        imageList[index], labelList[index] = imageTemp, labelTemp

                    # cv2.imwrite(os.path.join(r".\temp\Flip", "flip"+str(item)+"_img"+str(index) + "_image.png"), imageTemp)
                    # cv2.imwrite(os.path.join(r".\temp\Flip", "flip"+str(item)+"_img"+str(index) + "_label.png"), labelTemp)
        # 图片源扩展则返回扩展后列表，图片增广则返回单图片元素，图片已经被过滤(num=0)则没有返回值
        # print(len(imageList))
        if self.is_expand:
            return (imageList, labelList)
        elif num!=0:
            return (imageList, labelList)
        else:
            return ([], []) #必须有返回值


class randRotate(object):
    """依概率随机旋转"""
    def __init__(self, angles, is_expand=True, mode=cv2.INTER_NEAREST):
        self.angles = angles
        self.is_expand = is_expand
        self.mode = mode

    def __call__(self, image, label):
        if not isinstance(image,list): #传入的image是一个以array为元素的列表
            image, label = [image], [label]
        imageList, labelList = image, label  # 需要保留原始图片集
        num = len(image)

        try:
            size = image[0].shape  # 当图片剪切过程中尺寸过小将会在resize前被剔除，此时list长度为0，需要处理该异常
        except:
            # print(len(image))
            pass

        for index in range(num):
            imageTemp, labelTemp = image[index], label[index]
            for item in self.angles:
                if random.random() > 0.4:
                    x0, y0 = size[0] // 2, size[1] // 2  # 以图像中心作为旋转中心
                    method = cv2.getRotationMatrix2D((x0, y0), item, 1.0)  # 默认黑色填充边缘，获取旋转矩阵
                    imageTemp = cv2.warpAffine(imageTemp, method, size, borderMode =self.mode, borderValue=0)
                    labelTemp = cv2.warpAffine(labelTemp, method, size, borderMode =self.mode, borderValue=0)
                    # print(3, list(set(labelTemp.flatten().tolist())))

                    if self.is_expand:
                        imageList.append(imageTemp)
                        labelList.append(labelTemp)
                        imageTemp, labelTemp = image[index], label[index]  # 保证变化之间相互独立，即各种反转变化只进行一次
                    else:
                        imageList[index], labelList[index] = imageTemp, labelTemp  # 对原始图片连续变化
                    # cv2.imwrite(os.path.join(r".\temp\Rotate","rotate"+str(item)+"_img"+str(index) + "_image.png"), imageTemp)
                    # cv2.imwrite(os.path.join(r".\temp\Rotate","rotate"+str(item)+"_img"+str(index) + "_label.png"), labelTemp)
        # print(len(imageList))
        if self.is_expand:
            return (imageList, labelList)
        elif num!=0:
            return (imageList, labelList)
        else:
            return ([], []) #必须有返回值


class normalizationPro(object):
    """对每张图片归一化处理，对每张标签进行映射处理
    Deal with "DE", need to give meanStr"""
    def __init__(self, meanStr, labelMap, is_expand=True):
        self.mean, self.str = meanStr[0], meanStr[1]
        self.labelMap = labelMap
        self.is_expand = is_expand

    def __call__(self, image, label):
        if not isinstance(image,list): #传入的image是一个以array为元素的列表
            image, label = [image], [label]
        num = len(image)

        for index in range(num):
            image[index] = (image[index]-self.mean)/self.str #广播特性
            image[index] = image[index].astype(np.float32) #转为float32格式节省内存
            label[index][label[index]==200.] = self.labelMap[200.]
            label[index][label[index] == 500.] = self.labelMap[500.]
            label[index][label[index] == 600.] = self.labelMap[600.]
            label[index] = label[index].astype(np.uint8) #无符号整数节省内存
        # print(len(image))
        # return image, label if self.is_expand or num != 0 else [], []
        if self.is_expand:
            return (image, label)
        elif num != 0:
            return (image, label)
        else:
            return ([], [])  # 必须有返回值


class normalization(object):
    """对每张图片归一化处理，对每张标签进行映射处理
    Deal with subtype-list, calculate mean and str of each image, no need to give meanStr"""
    def __init__(self, labelMap, is_expand=True):
        self.labelMap = labelMap
        self.is_expand = is_expand

    def __call__(self, image, label):
        if not isinstance(image,list): #传入的image是一个以array为元素的列表
            image, label = [image], [label]
        num = len(image)

        for index in range(num):
            mean = (image[index]).mean()
            str = (image[index]).std()
            image[index] = (image[index]-mean)/str #广播特性
            image[index] = image[index].astype(np.float32) #转为float32格式节省内存
            label[index][label[index]==200.] = self.labelMap[200.]
            label[index][label[index] == 500.] = self.labelMap[500.]
            label[index][label[index] == 600.] = self.labelMap[600.]
            label[index] = label[index].astype(np.uint8) #无符号整数节省内存
        # print(len(image))
        # return image, label if self.is_expand or num != 0 else [], []
        if self.is_expand:
            return (image, label)
        elif num != 0:
            return (image, label)
        else:
            return ([], [])  # 必须有返回值


class singCropResize(object):
    """
    效仿transform的RandomResizedCrop函数，对传入的图片进行随机的剪裁
    接受单张图片以及参数：size, 面积增缩比例scale, 长宽比ratio
    """
    def __init__(self, crop_size=(256,256), rate=(3/4,4/3), scale=(0.95,1.0), mode=cv2.INTER_NEAREST):
        self.crop_size = crop_size
        self.scale = scale
        self.rate = rate
        self.mode = mode #邻近插值，默认的双线性插值会改变图片数值大小

    def get_params(self, img, scale, rate):
        # 输入：img: Array, scale: List[float], ratio: List[float]
        # 输出：Tuple[int, int, int, int]:
        width, height = img.shape  # 原图宽高
        area = height * width  # 原图面积

        log_ratio = np.log(np.array(rate))
        for _ in range(10):
            target_area = area * np.random.uniform(scale[0], scale[1])  # 原图面积*设置范围内随机生成一个缩放比例=>目标切割面积
            aspect_ratio = np.exp(np.random.uniform(log_ratio[0], log_ratio[1]))  # 同样，在设置的宽高比范围内获取一个随机值作为 本次的 宽高比
            # 根据切割面积和切割比例，计算出切割区域的宽和高
            # 以下是 这个方程组的解: xy=a; x/y=r; 其中x和y代表宽高，a代表面积，r代表比例
            # x = \sqrt(a*r); y = \sqrt(a/r)
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            # 如果在原图的范围内，就得到了最终的切割范围, 否则，尝试10次后，跳出循环，进行中心切割
            if 0 < w <= width and 0 < h <= height:
                i = np.random.randint(0, height - h + 1)
                j = np.random.randint(0, width - w + 1)
                # print(1, h,w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(rate):
            w = width
            h = int(round(w / min(rate)))
        elif in_ratio > max(rate):
            h = height
            w = int(round(h * max(rate)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        # print(2, h, w)
        return i, j, h, w

    def __call__(self, image, label):
        crop_params = self.get_params(image, self.scale, self.rate)

        imageCur = Image.fromarray(image) # ndarray转为PIL
        labelCur = Image.fromarray(label)
        imageCur = np.asarray(tF.crop(imageCur, *crop_params))  # 进行剪切并转回array
        labelCur = np.asarray(tF.crop(labelCur, *crop_params))

        # print(imageCur.shape)
        if (imageCur.shape[0] >= self.crop_size[0]) and (imageCur.shape[1] >= self.crop_size[1]):
            imageCur = cv2.resize(imageCur, self.crop_size, interpolation=self.mode) #邻近插值，默认的双线性插值会改变图片数值大小
            labelCur = cv2.resize(labelCur, self.crop_size, interpolation=self.mode)
            if set(labelCur.flatten().tolist()) == {0}:  # 去掉没有心脏的图片，防止后续程序报错
                return [], []
            else:
                return [imageCur], [labelCur]
        else:
            return [],[] #必须保证有返回值


class toTensor(object):
    def __call__(self, image, label): #接受一个列表，全部转为tensor，需要注意如果列表长度为1需要插入空维度1以保留维度
        image = torch.from_numpy(np.array(image[0])) if len(image)!=1 else torch.from_numpy(image[0]).unsqueeze(0)
        return image, torch.from_numpy(label[0])


class SegCompose(object):
    """
    Composes several transforms together. 考虑到语义分割需要同时变换image和label，必须重新定义Compose方法
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    实际上，将对图片依次进行各组变化，但是由于各种方法中存在一边多的情况导致了结果图片数增多，
    要求每个方法定义时必须能够处理图片列表传入的情况；注意每次必须保留了原始图片
    """
    def __init__(self, transforms):
        self.transforms = transforms #如果需要传入参数，那么在定义transform类时需要在init中设置为self变量

    def __call__(self, img, label):
        # transforms之间要注意上下游的输入与输出的格式保持不变。
        for t in self.transforms:
            img, label = t(img, label)
        return img, label


def readNPYImages_Pro(subType, npyPath=os.path.join(os.getcwd(), "DataInit"), is_train=True, kNum=1, defaultPro=None):
    """(Only deal with 1-subtype)
    读取npyPath下的npy文件，返回list
    defaultPro一般默认为false，但是如果传入了预处理方法则需要实施在每张图片上
    """
    setName = "train"+str(kNum) if is_train else "valid"+str(kNum)
    npyPathImage = os.path.join(npyPath, subType, setName, "Images")
    npyPathLabel = os.path.join(npyPath, subType, setName, "Labels")

    features, labels = [], []
    for i, fName in enumerate(os.listdir(npyPathImage)):
        # os.path.join会自动跨平台处理路径，主要由自己写的路径引起错误
        feature = np.load(os.path.join(npyPathImage, fName))
        label = np.load(os.path.join(npyPathLabel, fName))
        if not defaultPro is None:
            (feature, label) = defaultPro(feature, label) #可能返回空，此时将会被忽略
        features = features + feature
        labels += label
    return features, labels


def readNPYImages_V1(subType, npyPath=os.path.join(os.getcwd(), "DataInit"), is_train=True, kNum=1, defaultPro=None):
    """
    读取npyPath下的npy文件
    defaultPro一般默认为false，但是如果传入了预处理方法则需要实施在每张图片上
    """
    setName = "train"+str(kNum) if is_train else "valid"+str(kNum)
    features, labels = [], []
    if not isinstance(subType, list): #deal with 1-subtype
        subType = [subType]

    for item in subType:
        npyPathImage = os.path.join(npyPath, item, setName, "Images")
        npyPathLabel = os.path.join(npyPath, item, setName, "Labels")

        for i, fName in enumerate(os.listdir(npyPathImage)):
            # os.path.join会自动跨平台处理路径，主要由自己写的路径引起错误
            feature = np.load(os.path.join(npyPathImage, fName))
            label = np.load(os.path.join(npyPathLabel, fName))
            if not defaultPro is None:
                (feature, label) = defaultPro(feature, label) #可能返回空，此时将会被忽略
            features = features + feature
            labels += label

    return features, labels


def cur_pos_dic(npyPath=None):
    """Return dictionary of position information of given npyPath"""
    pos_sum_dic = {} #key: Int8: index of subject; value: Float32: index of pos in its subject
    for i, fName in enumerate(os.listdir(npyPath)):
        subj_index = fName.split("_")[0]
        # pos_index = int(fName.split("_")[2][:-4])
        # print(subj_index, pos_index)
        if subj_index in pos_sum_dic:
            pos_sum_dic[subj_index] += 1
        else:
            pos_sum_dic[subj_index] = 1
    return pos_sum_dic


def readNPYImages(subType, npyPath=os.path.join(os.getcwd(), "DataInit"),
                  is_train=True, kNum=1, defaultPro=None):
    """
    读取npyPath下的npy文件
    defaultPro一般默认为false，但是如果传入了预处理方法则需要实施在每张图片上(Also deal with defaultPro=None)
    Compare to V1: 3 return value, pos_list includes golden spatial array used in SCNN
                Can deal with visualization on test set(return current path)
    """
    if is_train: #Train or valid
        setName = "train"+str(kNum) if is_train else "valid"+str(kNum)
    else: #Test
        setName = "test"
    features, labels, pos_list, path_list = [], [], [], []
    if not isinstance(subType, list): #deal with 1-subtype
        subType = [subType]

    for item in subType:
        npyPathImage = os.path.join(npyPath, item, setName, "Images")
        npyPathLabel = os.path.join(npyPath, item, setName, "Labels")
        pos_sum_dic = cur_pos_dic(npyPathImage)

        for i, fName in enumerate(os.listdir(npyPathImage)):
            # os.path.join会自动跨平台处理路径，主要由自己写的路径引起错误
            feature = np.load(os.path.join(npyPathImage, fName))
            label = np.load(os.path.join(npyPathLabel, fName))
            if not defaultPro is None:
                (feature, label) = defaultPro(feature, label) #可能返回空，此时将会被忽略
                features = features + feature
                labels += label
            else:
                features = features + [feature] #Deal with defaultPro=None
                labels += [label]
            # Index Position of image
            gold_num_pos = pos_sum_dic[fName.split("_")[0]]
            gold_pos = (np.float32(fName.split("_")[2][:-4])+1) / gold_num_pos
            pos_list += [gold_pos] # List(Float32: 0~1) of gold position
            # Path name of image
            path_list += [fName]

    if is_train:
        return features, labels, pos_list
    else:
        return features, labels, path_list


class NPYSegDataset(torch.utils.data.Dataset):
    """
    一个用于加载npy数据集的自定义数据集
    数据结构：#第0维：类对象，第1维：二元元组, 第2维：71(相当于对1个图片增广成71张)个array构成的列表，最后两维：w*h
    在被调用时将会触动__getitem__方法，此时对image-label对逐个进行transform变换
    """
    def __init__(self, is_train=True, subType="DE", kNum=1):
        self.angles = [90, 180, 270] #旋转变化不能涉及中间值，因为通过旋转矩阵产生，会有新的数值导致label没有意义
        self.scaleRate = [0.8, 0.9, 0.95, 1] #用来保留原始图片,存储尺度缩减比例
        self.scale = [3/4, 4/3] #存储面积增缩比例范围
        self.rate = [0.8, 1] #存储变化后的长宽比
        self.flipCodes = [1, 0, -1]
        self.crop_size = (240, 240) #舍弃过小的图片，必须确保变化以后的尺寸为240*240
        self.subType = subType
        self.c0_lge_t2_mean_std = {"C0": (398.816, 395.903), "DE": (242.600, 158.449), "T2": (164.044, 182.646)}  # 均值和标准差
        self.labelMap = {200.: 1, 500.: 2, 600.: 3}
        self.is_expand = False
        self.interpolateMode = cv2.INTER_NEAREST
        self.is_train = is_train
        self.kNum = kNum
        # self.features, self.labels = self.applyTransforms() #调用transform方法并且串接成一个列表

        if is_train: #Train and valid
            self.transform = SegCompose([randFlip(self.flipCodes, self.is_expand),
                       randRotate(self.angles, self.is_expand, mode = self.interpolateMode),
                       normalization(self.labelMap, self.is_expand), #图片标准化
                       toTensor()
                        ])

            self.features, self.labels, self.gold_pos = readNPYImages(
                subType=self.subType, is_train=self.is_train, kNum=self.kNum,
                defaultPro=SegCompose([singCropResize(self.crop_size, self.rate, self.scale,
                                                      mode=self.interpolateMode)])
            )

        else: #Test
            self.transform = SegCompose([normalization(self.labelMap, self.is_expand), #图片标准化
                                         toTensor()
                                         ])

            self.features, self.labels, self.paths = readNPYImages(
                subType=self.subType, is_train=self.is_train, kNum=self.kNum,
                defaultPro=SegCompose([singCropResize(self.crop_size, self.rate, self.scale,
                                                      mode=self.interpolateMode)])
            ) #Different return value in test


    def __getitem__(self, idx):
        """Include the self.gold_pos needed for SCNN"""
        #通过索引对每个元素进行变化，相当于apply，在遍历数据时将会自动触发该函数
        # 需要注意的是dataloader要求每个getitem返回值相等，所以如果剪切后小于256*256被筛掉必须在init中进行
        feature, label = self.transform(self.features[idx], self.labels[idx])
        if self.is_train:
            gold_pos = self.gold_pos[idx]
            return (feature, label, gold_pos)
        else: #Different return value in test
            curpath = self.paths[idx]
            return (feature, label, curpath)

    """
    def __getitem__(self, idx):
        #Don't include the self.gold_pos needed for SCNN
        #通过索引对每个元素进行变化，相当于apply，在遍历数据时将会自动触发该函数
        # 需要注意的是dataloader要求每个getitem返回值相等，所以如果剪切后小于256*256被筛掉必须在init中进行
        feature, label = self.transform(self.features[idx], self.labels[idx])
        return (feature, label)
    """


    def __len__(self):
        return len(self.features)


# Deal with the case: getitem of DataLoader return different num of images 
def listPartition(init_list, prev_list, childern_list_len):
    '''
    init_list为初始化的列表2tuple*190(存在余数进行分析)image_num*256*256，prev_list来自上一个数据的余数，
    childern_list_len小列表长度，返回整数部分和余数部分
    需要注意的是最后返回的数值就是torch格式：batch_num * 2 * batch_size * w * h，余数：2 * 余数 * w * h
    '''
    imageList, labelList = init_list[0], init_list[1] #190image_num*256*256
    imageGroup = zip(*(iter(imageList),) *childern_list_len)
    labelGroup = zip(*(iter(labelList),) *childern_list_len)
    #存储拼接后结果，注意读取是双元素元组，存回来的还是双元素元组；
    # 如果不加zip，则会将两个zip拼接成一个然后每次读取相邻两个;
    intg_list = [(list(i), list(j)) for i, j in zip(imageGroup, labelGroup)] #23*2*batch_size*256*256
    if not prev_list: #说明上一次没有余数
        prev_list = ([], [])
        count = len(imageList) % childern_list_len
    else:
        count = (len(imageList) + len(prev_list[0])) % childern_list_len  # 处理余数6*batch_size*256*256(同时引入上一次的余数部分)

    if count != 0:
        if count >= childern_list_len: #说明引入上一次结果后可以进行再次处理
            imageRemList, labelRemList = imageList[-count:]+prev_list[0], labelList[-count:]+prev_list[1]
            imageRemGroup = zip(*(iter(imageRemList),) * childern_list_len)
            labelRemGroup = zip(*(iter(labelRemList),) * childern_list_len)
            intg_list.append([(list(i), list(j)) for i, j in zip(imageRemGroup, labelRemGroup)])
            count1 = count % childern_list_len #计算当下的剩余
            if count1 != 0:
                rem_list = (imageRemList[-count1:], labelRemList[-count1:]) #说明拼接以后恰好整除了
            else:
                rem_list = False
        else:
            rem_list = (imageList[-count:], labelList[-count:]) #2Tuple*6List*batch_size*256*256
    else:
        rem_list = False
    # torch.tensor主要是为了将原来List格式的图片转为tensor，最后一维再进行变换，注意将元组维度切换到batchsize维度前面
    # print(torch.tensor(intg_list).shape)
    #list直接转为tensor会消耗大量时间
    return torch.tensor(np.array(intg_list)).transpose(1,0), rem_list #2*23*batch_size*256*256,transpose()只能一次操作两个维度


def collate_fn(batch):
#
#     原文链接：https://blog.csdn.net/qq_46092061/article/details/119348106
#     batch是一个数据元组，元组长度就是你的batch_size设置的大小,每个元组是dataset中_getitem__的结果：tuple2(list192(ndarray(256*256)))
#     如果不指定该函数则采用使用DataLoader默认的处理方式，此时要求必须每个batch的规模相等，相当于对8组tuple2(darray(192*256*256))拼接
#     我们希望返回的结果是24batch_num * 2 *torch.Size([8, 256, 256])，即在一组的内部进行处理，这样可以支持第一个数字未必相等
#     此时每次抛出为2 *torch.Size([8, 256, 256])， 第一维batch_num作为迭代器抛出的外循环

    batch_size = len(batch) #batch_size*2*192*256*256
    # print(batch_size, len(batch[0]), len(batch[0][0]))
    # print(type(batch), type(batch[0]), type(batch[0][0]))
    remTupList = False
    resTensor = torch.tensor([])
    for i in range(batch_size): #2*192*256*256
        # if isinstance(remTupList,tuple): # 拼接列表2*192*256*256与2*0*256*256
        #     curList = [0, 0]
        #     curList[0] = batch[i][0] + remTupList[0]
        #     curList[1] = batch[i][1] + remTupList[1]
        #     curList = tuple(curList)
        # else:
        #     curList = batch[i]
        curList = batch[i]
        #传入2*192*256*256, 返回2*24*batch_size*256*256, 余量2*0*256*256
        intgTensor, remTupList = listPartition(curList, remTupList, batch_size) #将上一次的剩余放到下一次处理
        resTensor = torch.cat([resTensor, intgTensor], dim=1) if i != 0 else intgTensor  # 在第1维(batch_num)堆积

    del batch
    # print(resTensor.shape) #实际上在batch操作时不对图片Array(256*256)操作
    return resTensor #tensor(batch_size*batch_num*2*256*256)





"""
#测试dataloader处理不同数目的getitem的图片组

myData = NPYSegDataset() #第0维：类对象，第1维：二元元组, 第2维：71个array构成的列表，最后两维：w*h
# print(len(myData[0]), len(myData[0][0]), myData[0][0][0].shape) #数据被调用时将会触发get_item方法

time_start=time.time()
trainData = NPYSegDataset()
batch_size = 8

time_start1=time.time()

# 如果is_expand指定为真则最后每个batch的图片数随机，需要指定collate_fn来处理
# train_iter = torch.utils.data.DataLoader(trainData, batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
train_iter = torch.utils.data.DataLoader(trainData, batch_size, shuffle=True, drop_last=True) #由于剪裁结果一定相等，不需要指定collate_fn

num = 0
for X, Y in train_iter:
    print(len(X), X[0].shape, type(X[0][0][0])) #192 torch.Size([8, 256, 256])，只需要修正train函数读取时的代码即可，加一层循环
    print(len(Y), Y[0].shape, Y[0])
    # num+=1
    break
# print(num)

time_end=time.time()
print('time cost',time_start1-time_start,'s\n')
print('time cost',time_end-time_start1,'s')
"""



"""
#测试重写的transform方法和compose
img = np.load(r".\DataInit\C0\train0\Images\subject4_C0_5.npy")
label = np.load(r".\DataInit\C0\train0\Labels\subject4_C0_5.npy")
angles = [45, 90, 135, 180, 225, 270, 315]
scaleRate = [0.8, 0.85, 0.9, 0.95, 1]
flipCodes = [1, 0, -1]
c0_lge_t2_mean_std = {"C0":(398.816, 395.903), "DE":(242.600, 158.449), "T2":(164.044, 182.646)} #均值和标准差
labelMap = {200.: 1, 500.:2, 600.:3}

# myTransforms = SegCompose([randCropResize(scaleRate, crop_size=(256,256))])
myTransforms = SegCompose([randCropResize(scaleRate, crop_size=(256,256)),
                           randFlip(flipCodes, is_expand=False),
                           randRotate(angles, is_expand=False),
                            normalization(c0_lge_t2_mean_std["C0"], labelMap, is_expand=False)
                           ])

a = myTransforms(img, label) #0：二维元组；1：图片列表
# print(len(a), len(a[0]), type(a[0])) #测试图片增广
print(len(a), len(a[0]), a[0][0].shape, type(a[0][0])) #测试图片扩展2 2 (256, 256) <class 'numpy.ndarray'>
"""