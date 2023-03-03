"""
@author: Menghan Qin
call by Train(net, net_name, train_iter, test_iter, loss_type, acc_type, animator_type,
    trainer, num_epochs, devices)
"""
import os.path

import d2l.torch as d2l
import torch
from torch import nn
from core.AccuracyMetrix import AccuracyCompose as accuracy
from core.LossFunction import CrossCompose as loss_fun
import numpy as np
from IPython import display
from matplotlib_inline import backend_inline
from matplotlib import pyplot as plt


class Accumulator:
    """For accumulating sums over `n` variables. Modified on d2l.Accumulator,
    but can accept tensor as input of add."""
    def __init__(self, n):
        self.n = n
        self.data = [0.0] * n

    def add(self, *args):
        # You can deal with more data-type by adding in the try-part or except-part depending on the situation
        num = 0  # counter
        for item in args:
            try:
                if item.grad_fn is not None:
                    self.data[num] += float(item.data)  # deal with loss: tensor(40.1641, grad_fn=<SumBackward0>)
                    num += 1
                elif torch.is_tensor(item):
                    for item_num in range(item.shape[0]):
                        self.data[num] += float(item[item_num])
                        num += 1
            except:
                if isinstance(item,list):
                    for i in item:
                        self.data[num] += float(i)
                        num += 1
                else:
                    self.data[num] += float(item)
                    num += 1
        # print(self.data)

    def reset(self):
        self.data = [0.0] * self.n

    def __getitem__(self, idx):
        return self.data[idx]


class Animator:
    """For plotting data in animation. Modified on d2l.Animator: Incrementally plot multiple lines
    Add data by AnimatorObject.add(xCoordinate, yCoordinate, row, col):share same configues(or use  list-index)
    Use by AnimatorObject.show_axes():update at one time
    Modify set_axes to change config of sub-images
    !!!Note: 2 MODES: subplots whether to share same configues(yes)
                whether to update each time during data-adding  or just at one time(one time)"""
    def __init__(self, fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1, figsize=(9, 6),
                 xlabel=None, ylabel=None, xlim=None, ylim=None, title=None,
                 xscale='linear', yscale='linear', legend=None):
        # nrows: rows of subplot; ncols: cols of subplot
        # Use the svg format to display a plot in Jupyter.
        backend_inline.set_matplotlib_formats('svg')
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize) #Return nr*nc Subplot
        if nrows * ncols == 1:
            self.axes = [[self.axes, ], ] #Deal with 1*1Plot
        elif nrows==1 and ncols!=1: #Exits some problems when deal with 1*2 or 2*1 subplot
            self.axes = [self.axes, ]
        # save configue
        self.nrows, self.ncols = nrows, ncols
        self.X, self.Y, self.fmts = None, None, fmts

        self.xlabel, self.ylabel = xlabel, ylabel
        self.xlim, self.ylim = xlim, ylim
        self.xscale, self.yscale = xscale, yscale
        self.legend, self.title = legend, title

    def add(self, x, y, row=0, col=0):
        # Add multiple data points into the figure[row][col]
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[row][col].cla() #cla清理当前的axes
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[row][col].plot(x, y, fmt)    #axes[row][col]
        # #capture arguments(given axe, with same attributes)
        # self.set_axes(self.axes[row][col], self.xlabel, self.ylabel, self.xlim,
        #          self.ylim, self.xscale, self.yscale, self.legend)
        # # Show images
        # display.display(self.fig)
        # display.clear_output(wait=True)

    def reset_add(self):
        """set X, Y to be None"""
        self.X, self.Y = None, None

    def set_axes(self, axes, xlabel=None, ylabel=None, xlim=None, title="kNum=0",
                 ylim=None, xscale='linear', yscale='linear', legend=None, cur_title=None):
        """Set the axes for matplotlib. Modified on d2l.set_axes
        axes accept one figure"""
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_xscale(xscale)
        axes.set_yscale(yscale)
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)
        if title!=None:
            axes.set_title(cur_title)
        if legend!=None:
            axes.legend(legend)
        axes.grid()

    def show_axes(self, foldername, filename, row=0, col=0):
        # capture arguments(given axe, with same attributes)

        self.set_axes(axes=self.axes[row][col], xlabel=self.xlabel, ylabel=self.ylabel,
                      xlim=self.xlim, ylim=self.ylim, xscale=self.xscale, yscale=self.yscale,
                      legend=self.legend)
        # Show and save images
        plt.savefig(os.path.join(foldername, filename)) #save Figure-Object in python
        # display.display(self.fig) #show in jupyter
        display.clear_output(wait=True)


def try_all_gpus():  #@save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


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


class Train(object):
    def __init__(self, net, net_name = "U-Net", train_iter=None, test_iter=None, is_pretrain_SR=False,
                 loss_type="lossCrossEntropy", regular_type=None, SR_path=None, use_SC=False, use_SR=False,
                 acc_type=["dice", "Average Symmetric Surface Diatance","Hausdorff distance","correct prediction of pixel"],
                 animator_type="dice", trainer=None, num_epochs=30, devices="cuda",
                 type_num=4, kNum=0, Animator=None, row=0, col=0,
                 foldername=os.path.join(os.getcwd(), "Result"), filename="print_message1.txt"):
        """
        :param net:
        :param net_name:
        :param train_iter:
        :param test_iter:
        :param is_pretrain_SR：whether to pretrain SR parameters(gold standard to be both images and labels)
        :param loss_type: Str: type name of loss term
        :param regular_type: Str: type name of regular term
        :param SR_path: Str: Load path of SR Parameters
        :param use_SC: Bool: whether to use SC regular term(add out_SC to regular_term, else None)
        :param use_SR: Bool: whether to use SR regular term(change images input from 1 channel to 4-onehots)
        :param acc_type: Str-List: all type name of accuracy needed to print finally
        :param animator_type: Str: type of accuracy needed to show in figures
        :param trainer:
        :param num_epochs:
        :param devices:
        :param type_num:
        :param kNum: Int: index of Cross Validation
        :param Animator:
        :param row: Int: num of row for Animator
        :param col: Int: num of colume for Animator
        :param foldername: Str: save Folder path
        :param filename: Str: save filename of txt-file
        """
        self.net = net
        self.net_name = net_name
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.loss_type = loss_type
        self.regular_type = regular_type
        self.use_SC = use_SC
        self.use_SR = use_SR
        self.SR_path = SR_path
        self.is_pretrain_SR = is_pretrain_SR
        self.type_num = 4
        self.acc_type = acc_type
        self.animator_type = animator_type #the accuracy we show on picture
        self.trainer = trainer
        self.num_epochs = num_epochs
        self.devices = devices
        self.Accumulator = Accumulator
        self.Animator = Animator
        self.Animator.reset_add() #Reset X_Content and Y_Content to be None each KNum
        self.kNum = kNum
        self.row, self.col = row, col
        self.foldername = foldername #path to save txt-file
        self.filename = filename

    def labels_to_onehot(self, target, type_num):
        """change target from: Int(0,1,2,3) to: Typenum(binary)"""
        shape = target.shape
        onehot_tensor = torch.zeros((shape[0], type_num, shape[1], shape[2])).to(self.devices)
        for type_index in range(type_num):
            onehot_tensor[:, type_index] = (target==type_index).to(torch.float32)
        return onehot_tensor


    def train_batch(self, X, y, gold_pos):
        """Train for a minibatch with mutiple GPUs, modified on d2l.train_batch_ch13"""
        if isinstance(X, list):
            # Required for BERT fine-tuning (to be covered later)
            X = [x.to(self.devices) for x in X]
        else:
            X = X.to(self.devices)
        y = y.to(self.devices)
        gold_pos = gold_pos.to(self.devices)
        self.net.train()
        self.trainer.zero_grad()
        if self.use_SC:
            pred, pos_out = self.net(X)
        else:
            pred = self.net(X)
            pos_out = None
        # loss batch-sum
        loss_object = loss_fun(pred, y, self.loss_type, self.regular_type, device=self.devices,
                               SR_path=self.SR_path, use_SR=self.use_SR, pos_out=pos_out, gold_pos=gold_pos)
        l, percentage = loss_object.loss_function() #percentage means loss_term/(loss_term+regular_term)
        l.sum().backward()
        self.trainer.step()
        train_loss_sum = l.sum()
        # accuracy tensor
        with torch.no_grad():
            accuracy_object = accuracy(pred, y, self.acc_type, devices=self.devices)
            train_acc_tensor, train_num_tensor = torch.tensor(accuracy_object.get_all_acc())
        return train_loss_sum, percentage, train_acc_tensor, train_num_tensor


    def train(self):
        """Train a model with mutiple GPUs, modified on d2l.train_ch13"""
        animator_index = self.acc_type.index(self.animator_type) #the accuracy shown in picture
        timer, num_batches = d2l.Timer(), len(self.train_iter)
        # animator = self.Animator(xlabel='epoch', xlim=[1, self.num_epochs], ylim=[0, 1],
        #                         legend=['train loss', 'train acc', 'test acc'])
        # self.net = nn.DataParallel(self.net, device_ids=self.devices).to(self.devices) #Train on multi-GPU
        self.net.to(self.devices) #Train on single GPU
        str1 = ("Cross Validation: No.%d"%(self.kNum)).center(70, "*") #Print Dividing line
        print(str1)
        if self.kNum==0:
            write_line(foldername=self.foldername, filename=self.filename,
                       str=str1, reset=True)
        else:
            write_line(foldername=self.foldername, filename=self.filename,
                       str=str1, reset=False)

        for epoch in range(self.num_epochs):
            # Sum of training loss, sum of training accuracy, no. of examples,
            # no. of predictions
            metric = self.Accumulator(3+1+len(self.acc_type))
            for i, (features, labels, gold_pos) in enumerate(self.train_iter):
                timer.start() #function timer of self.train_batch()
                if self.is_pretrain_SR: # gold labels to be both images(one-hot code) and labels
                    label_onehot = self.labels_to_onehot(labels, type_num=self.type_num)
                    l, percentage, acc_tensor, num_tensor = self.train_batch(label_onehot, labels, gold_pos)
                else:
                    l, percentage, acc_tensor, num_tensor = self.train_batch(features, labels, gold_pos)
                metric.add(l, percentage, acc_tensor, labels.shape[0], num_tensor[animator_index])
                timer.stop()
                # if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1: #print per 0.2epoch
                #     animator.add(epoch + (i + 1) / num_batches, #animator of train loss and train acc
                #                  (metric[0] / metric[-2], metric[animator_index+1] / metric[-1],
                #                   None))
                if i == num_batches-1:
                    self.Animator.add(epoch+1, (metric[0] / metric[-2],
                                metric[animator_index+1+1] / metric[-1], None), row=self.row, col=self.col)
                    str1 = "Epoch: %d; Train Loss: %.3f; Loss Percentage: %.3f; Train Acc: %.3f;"%(epoch+1,
                                metric[0] / metric[-2], metric[1] / metric[-2],
                                metric[animator_index+1+1] / metric[-1])

            test_acc = self.evaluate_accuracy_gpu(data_iter = self.test_iter) #traverse test-set
            self.Animator.add(epoch + 1, (None, None, test_acc[animator_index]),
                              row=self.row, col=self.col)  # animator of test acc

            str1 += " Test Acc: %.3f"%(test_acc[animator_index])
            print(str1) #print in console
            write_line(foldername=self.foldername, filename=self.filename,
                       str=str1, reset=False) #save in txt-file

        str1 = "\n\n"+("All Accuracy of kNum: %d"%(self.kNum)).center(70, "*")
        print(str1)
        write_line(foldername=self.foldername, filename=self.filename,
                   str=str1, reset=False)

        str_res = "loss %.3f, " % (metric[0] / metric[-2])
        for acc_index in range(len(self.acc_type)): #concat the result to print
            str_res += "train "+self.acc_type[acc_index]+" %.3f, " % (metric[1+1+acc_index] / metric[-1])
            str_res += "test " + self.acc_type[acc_index] + " %.3f, " % (test_acc[acc_index]) + "\n"
        print(str_res)
        write_line(foldername=self.foldername, filename=self.filename,
                   str=str_res, reset=False)

        print(f'{metric[-2] * self.num_epochs / timer.sum():.1f} examples/sec on '
              f'{str(self.devices)}'+"\n\n")
        write_line(foldername=self.foldername, filename=self.filename,
                   str=f'{metric[-2] * self.num_epochs / timer.sum():.1f} examples/sec on '
              f'{str(self.devices)}'+"\n\n", reset=False)




    def evaluate_accuracy_gpu(self, data_iter):
        """Compute all accuracy`in acc_type, and average them on dataset
        Return: List: acc_type_num;   modified on d2l.evaluate_accuracy_gpu"""
        if isinstance(self.net, torch.nn.Module):
            self.net.eval()  # Set the model to evaluation mode
            if not self.devices:
                device = next(iter(self.net.parameters())).self.devices
        # metric: predictions, metric_num: num
        metric = self.Accumulator(len(self.acc_type))
        metric_num = self.Accumulator(len(self.acc_type))

        with torch.no_grad():
            for X, y, _ in data_iter:
                if isinstance(X, list):
                    # Required for BERT Fine-tuning (to be covered later)
                    X = [x.to(self.devices) for x in X]
                else:
                    X = X.to(self.devices)
                y = y.to(self.devices)
                if self.is_pretrain_SR:
                    X1 = self.labels_to_onehot(y, type_num=self.type_num)
                else:
                    X1 = X

                if self.use_SC:
                    accuracy_object = accuracy(self.net(X1)[0], y, self.acc_type, devices=self.devices)
                else:
                    accuracy_object = accuracy(self.net(X1), y, self.acc_type, devices=self.devices)
                acc_return = torch.tensor(accuracy_object.get_all_acc()[0])
                num_return = torch.tensor(accuracy_object.get_all_acc()[1])
                metric.add(acc_return)
                metric_num.add(num_return)
        return [a / (b+1e-5) for a, b in zip(metric, metric_num)]

