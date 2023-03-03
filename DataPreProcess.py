"""
@author: Menghan Qin
数据结构：mscmr_image中存储着原始数据，mscmr_manual中存储着手动标定结果，需要进行匹配建立Image-Label对
获取K则交叉验证的文本文件
将labels存储为png格式(0-255)彩色图片用于观察，使用npy保存原始的ndarray数据格式(几万)
"""

import nibabel as nib
import cv2
import os
import numpy as np
from tqdm import tqdm
import random
from DataLoader import randCropResize, randFlip, randRotate, SegCompose, normalization #将所有图片进行增广处理并且存储到本地文件
# (导入时注意注释原文件代码)

# np.set_printoptions(threshold=np.inf) # print all data
typeList = ["C0","DE", "T2"]

def searchPath(path):
    """Creat folder if given path doesn't exist"""
    if not os.path.isdir(path):
        os.makedirs(path)


def writeTxt(path, lst):
    """write each item of list to txt-file(given path), judge whether to override(at first time)
    For more: https://www.cnblogs.com/ymjyqsx/p/6554817.html"""
    writeMode = "w" # override txt-file(creat automatically)
    for item in lst:
        with open(path, writeMode) as f:
            f.writelines(item + "\n")  # write each line
        if os.path.getsize(path):  # judge txt-file whether to be None
            writeMode = "a" # only continue writing(creat automatically)


def fFoldTxt(initPath, savePath, k=6):
    """
    建立交叉验证所需要的文本文件：先随机选取15个作为测试集，剩余30个取6择交叉验证
    将文件名(去除后缀)存储进6*2+1个文本文件，k为K叉验证，initPath为数据原始存储路径，savePath为最终存储路径
    """
    imgPathListAll = os.listdir(initPath)  # 返回图片组下的所有子图片路径列表
    for subType in typeList: #处理三个模态的图片
        imgPathList = list(filter(lambda item: subType in item, imgPathListAll)) #过滤对应模态图片
        searchPath(os.path.join(savePath, subType)) #创建存放模态信息的文件夹
        # imgPathList=list(filter(lambda item: "DE" in item, imgPathList))
        # print(len(imgPathList))

        tempIndice = list(range(15)) #需要修正：对45进行随机变化，选取前15个作为test
        random.shuffle(tempIndice)  # 产生0~num-1的随机序列用作索引，以进行随机划分(没有返回值)
        imgPathList = np.array(imgPathList)  # 列表不可以使用列表作为索引，但是Array可以

        testPathList = imgPathList[tempIndice] #test的路径列表
        imgPathList = np.delete(imgPathList, tempIndice) #删除test对应项
        testPath = os.path.join(savePath, subType, "test.txt")
        writeTxt(testPath, testPathList)

        for i in range(k):
            #产生以第i块数据作为验证集的数据和标签
            trainPathList = None
            for j in range(k):
                idx = slice(j*5, (j+1)*5) #第j组切片索引号
                if j==i:
                    validPathList = imgPathList[idx]
                elif trainPathList is None: #第一次创建
                    trainPathList = imgPathList[idx]
                else:
                    trainPathList = np.append(trainPathList, imgPathList[idx])

            #至此获得了全部的路径列表，需要存储至txt文件夹中
            trainPath = os.path.join(savePath, subType, "train"+str(i)+".txt")
            validPath = os.path.join(savePath, subType, "valid" + str(i)+".txt")
            writeTxt(trainPath, trainPathList)
            writeTxt(validPath, validPathList)


def pngToRGB(savePath, img):
    # 将标签变为彩色图片进行存储，仅用于显示，实际分类时标签将200,500,600映射到0，1，2
    # 如果本身就是RGB图片更多则需要引入相关的处理函数处理三元元组的映射，但是此处的RGB仅用于显示而已
    savetemp = np.zeros((img.shape[0], img.shape[1], 3))  # 用于存储彩色图片
    # 布尔值矩阵（第一维B，第二维G，第三维R）对三通道依次变换
    # 200.:(69.97.143); 500.:(172,158,122); 600.:(212,133,175)
    Bool200 = (img == 200.)
    Bool500 = (img == 500.)
    Bool600 = (img == 600.)
    savetemp[:, :, 0] = (Bool200).astype(np.uint8) * 69 + (Bool500).astype(np.uint8) * 172 + (Bool600).astype(np.uint8) * 212
    savetemp[:, :, 1] = (Bool200).astype(np.uint8) * 97 + (Bool500).astype(np.uint8) * 158 + (Bool600).astype(np.uint8) * 133
    savetemp[:, :, 2] = (Bool200).astype(np.uint8) * 143 + (Bool500).astype(np.uint8) * 122 + (Bool600).astype(np.uint8) * 175
    cv2.imwrite(savePath, savetemp) #RGP图片类型本身就是0-255的unint8，不存在失真的情况


def arrayToPNG(savePath, img):
    """
    由于array数据格式是没有范围限制的float64类型，我们希望对他归一化并转为float32格式的灰度图片
    实际上png图片只是用于可视化，所以不需要变为PIL格式，直接cv2保存为灰度图片即可
    """
    maxPix = img.max().max()
    minPix = img.min().min()
    img = ((img-minPix)*255/(maxPix-minPix)).astype(np.uint8) #进行归一化并转为uint8格式
    # GrayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #相当于手动转为了灰度图，则不需要再转换
    cv2.imwrite(savePath, img)


def readNiiImages(initPath, txtPath, saveInitPath, is_train=True, kNum=1):
    """
    按照txt文件，读取对应的nii图片并解压，返回图片和标签
    initPath二元列表，存储原始的两个路径，txtPath存储txt的初始路径，savePath存储的初始路径
    目前对于kNum, subType, is_train进行循环遍历，实际上对于features, labels可以进行一并扩展
    """
    for kNum in range(6):
        for subType in typeList:
            for is_train in [False, True]:
                txtFname = os.path.join(txtPath, subType, #根据模式读取对应的txt文件
                                         'train%d.txt'%(kNum) if is_train else 'valid%d.txt'%(kNum))
                if is_train:
                    savePath = os.path.join(saveInitPath, subType, "train%d"%(kNum)) #此时仅考虑DE及train1
                else:
                    savePath = os.path.join(saveInitPath, subType, "valid%d"%(kNum))
                with open(txtFname, 'r') as f:
                    images = f.read().split() #打开txt文件，默认以空格作为切割符，获得字符串列表
                    # print(images[0])
                searchPath(os.path.join(savePath, "Images"))
                searchPath(os.path.join(savePath, "Labels"))

                features, labels = [], []
                for i, fname in enumerate(tqdm(images)):
                    img1 = nib.load(os.path.join(initPath[0], fname))
                    img2 = nib.load(os.path.join(initPath[1], fname.replace(".nii.gz", "_manual.nii.gz")))
                    for j in range(img1.shape[2]):
                        temp1 = img1.get_fdata()[:, :, j]  # 返回图片的浮点数格式(ndarray)
                        features.append(temp1) #对列表进行扩展
                        # 保存在.\DataPNG\subType\train1\Images\fname
                        # arrayToPNG(os.path.join(savePath, "Images", fname[:-7]+"_%d.png"%(j)), temp1)
                        # 保存在.\Data\subType\train1\Images\fname
                        np.save(os.path.join(savePath, "Images", fname[:-7]+"_%d.npy"%(j)), temp1)
                        temp2 = img2.get_fdata()[:, :, j]  # 返回图片的浮点数格式(ndarray)
                        labels.append(temp2)
                        # 保存在.\DataPNG\subType\train1\Labels\fname
                        # pngToRGB(os.path.join(savePath, "Labels", fname[:-7]+"_%d.png"%(j)), temp2)
                        # 保存在.\Data\subType\train1\Labels\fname
                        np.save(os.path.join(savePath, "Labels", fname[:-7]+"_%d.npy"%(j)), temp2)
    # return features, labels


def readTestNiiImages(initPath, txtPath, saveInitPath):
    """
    按照txt文件，读取对应的nii图片并解压，返回图片和标签
    initPath二元列表，存储原始的两个路径，txtPath存储txt的初始路径，savePath存储的初始路径
    目前对于subType进行循环遍历，生成测试集
    """
    for subType in typeList:
        txtFname = os.path.join(txtPath, subType, 'test.txt')#根据模式读取对应的txt文件
        savePath = os.path.join(saveInitPath, subType, 'test')
        with open(txtFname, 'r') as f:
            images = f.read().split() #打开txt文件，默认以空格作为切割符，获得字符串列表
            # print(images[0])
        searchPath(os.path.join(savePath, "Images"))
        searchPath(os.path.join(savePath, "Labels"))

        features, labels = [], []
        for i, fname in enumerate(tqdm(images)):
            img1 = nib.load(os.path.join(initPath[0], fname))
            img2 = nib.load(os.path.join(initPath[1], fname.replace(".nii.gz", "_manual.nii.gz")))
            for j in range(img1.shape[2]):
                temp1 = img1.get_fdata()[:, :, j]  # 返回图片的浮点数格式(ndarray)
                features.append(temp1) #对列表进行扩展
                # 保存在.\DataPNG\subType\test\Images\fname
                arrayToPNG(os.path.join(savePath, "Images", fname[:-7]+"_%d.png"%(j)), temp1)
                # 保存在.\Data\subType\test\Images\fname
                # np.save(os.path.join(savePath, "Images", fname[:-7]+"_%d.npy"%(j)), temp1)
                temp2 = img2.get_fdata()[:, :, j]  # 返回图片的浮点数格式(ndarray)
                labels.append(temp2)
                # 保存在.\DataPNG\subType\test\Labels\fname
                pngToRGB(os.path.join(savePath, "Labels", fname[:-7]+"_%d.png"%(j)), temp2)
                # 保存在.\Data\subType\test\Labels\fname
                # np.save(os.path.join(savePath, "Labels", fname[:-7]+"_%d.npy"%(j)), temp2)


def imagePreExpand(initPath, savePath):
    """将之前处理好的图片再应用图片数量扩展：剪切、归一化并存储到本地"""
    # subType = "DE"
    # kNum = 1
    # is_train = "train"
    for subType in typeList:
        for kNum in range(6):
            for is_train in ["train", "valid"]:
                setName = is_train+str(kNum)
                # setName = "test"
                initTempPathImg = os.path.join(initPath, subType, setName, "Images")
                saveTempPathImg = os.path.join(savePath, subType, setName, "Images")
                initTempPathLab = os.path.join(initPath, subType, setName, "Labels")
                saveTempPathLab = os.path.join(savePath, subType, setName, "Labels")
                searchPath(saveTempPathImg)
                searchPath(saveTempPathLab)

                angles = [45, 90, 135, 180, 225, 270, 315]
                scaleRate = [0.8, 0.9, 0.95, 1]  # 1用来保留原始图片
                flipCodes = [1, 0, -1]
                crop_size = (256, 256)  # 舍弃过小的图片，必须确保变化以后的尺寸为256*256
                c0_lge_t2_mean_std = {"C0": (398.816, 395.903), "DE": (242.600, 158.449),
                                      "T2": (164.044, 182.646)}  # 均值和标准差
                labelMap = {200.: 1, 500.: 2, 600.: 3}
                # transform = SegCompose([randCropResize(scaleRate, crop_size),  # 定义变换方法
                #                         randFlip(flipCodes),
                #                         randRotate(angles),
                #                         normalization(c0_lge_t2_mean_std[subType], labelMap)
                #                         ])
                transform = SegCompose([randCropResize(scaleRate, crop_size)])

                pathDirImg =  os.listdir(initTempPathImg) #img和label图片名称相同
                for item in tqdm(pathDirImg):
                    tempImg = np.load(os.path.join(initTempPathImg, item))
                    tempLab = np.load(os.path.join(initTempPathLab, item))

                    tempList = transform(tempImg, tempLab) #生成list192*ndarray(256*256)
                    for index in range(len(tempList[0])):
                        # np.save(os.path.join(saveTempPathImg, item[:-4]+"_"+str(index)+".npy"), tempList[0][index])
                        # np.save(os.path.join(saveTempPathLab, item[:-4] + "_" + str(index) + ".npy"), tempList[1][index])
                        arrayToPNG(os.path.join(saveTempPathImg, item[:-4]+"_"+str(index)+".png"), tempList[0][index])
                        pngToRGB(os.path.join(saveTempPathLab, item[:-4] + "_" + str(index) + ".png"), tempList[1][index])



# fFoldTxt(".\mscmr_image", ".\kFolder") #创建.\kFold\subType\.txt文件
# readNiiImages([".\mscmr_image", ".\mscmr_manual"], ".\kFolder", ".\DataInit", is_train=True) #解压nii文件，并对结果存储到List中
readTestNiiImages([".\mscmr_image", ".\mscmr_manual"], ".\kFolder", ".\DataPNG") #处理测试集
# imagePreExpand(initPath=r".\DataInit", savePath=r".\DataPNG")


"""
# 通过遍历所有手动标记的压缩文件中的每个indice，找出所有出现的值
#观察数据可知数据点为200.，500.,600.，数据类型为float64
np.set_printoptions(threshold=np.inf) #打印完整格式

pathlist = os.listdir(r".\mscmr_manual")
templist = [] #用来保存所有值
for item in tqdm(pathlist):
    img = nib.load(os.path.join(r".\mscmr_manual", item))
    # print(img.shape) #w*h*num
    for i in range(img.shape[2]):
        temp = img.get_fdata()[:,:,i] #返回图片的浮点数格式(ndarray)
        templist = list(set(templist+temp.flatten().tolist())) #找出不重复值
print(templist)
"""