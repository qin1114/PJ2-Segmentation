import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict


def bilinear_kernel(in_channels, out_channels, kernel_size): #双线性插值采样
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    for i in range(in_channels):
        weight[i, range(out_channels), :, :] = filt
    return weight


"""SCNN Part: Support basic UNet and SCNN"""
class EncoderBlock_SC(nn.Module):
    """2 3*3conv + 1 2*2Maxpool or 2*2up-conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1, padding=1,
                 is_final=False):
        super(EncoderBlock_SC, self).__init__()
        if not is_final:  # whether to be the Transfer-Block(with no Maxpool)
            final_layer = nn.MaxPool2d(2, 2)
            self.final_name = "Maxpool"
        else:  # use bias
            # self.W_final = bilinear_kernel(out_channels, in_channels, kernel_size=2)
            final_layer = nn.ConvTranspose2d(out_channels, in_channels, kernel_size=2, stride=2)
            self.final_name = "trans_conv"
            # self.final_layer.weight.data.copy_(self.W_final)  # bilinear_kernel Init

        self.main = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                stride=strides, padding=padding, bias=False)),
            ('bn1', nn.BatchNorm2d(out_channels)),
            ('conv2', nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                                stride=strides, padding=padding, bias=False)),
            ('bn2', nn.BatchNorm2d(out_channels)),
            (self.final_name, final_layer)
        ]))

    def forward(self, x):
        # Able to Deal with SC, since SC need the saveout of trans_conv
        out = x
        for name, module in self.main.named_children():
            out = module(out)
            # print(name, module, out.shape)
            if "bn" in name:  # Add Relu after bn layer
                out = F.relu(out)
            if name == "bn2":
                out_save = out

        return out, out_save

    """
        def forward(self, x):
    # Deal with no_SC version, since SC need the saveout of trans_conv
        out = x
        for name, module in self.main.named_children():
            out = module(out)
            # print(name, module, out.shape)
            if "bn" in name: #Add Relu after bn layer
                out = F.relu(out)
            if name=="bn2" and self.final_name!="trans_conv":
                out_save = out

        if self.final_name!="trans_conv":
            return out, out_save
        else:
            return out
    """


class DecoderBlock_SC(nn.Module):
    """2 3*3conv + 1 2*2up-conv or 1*1conv"""

    def __init__(self, in_channels, mid_channels, out_channels, kernel_size=3, strides=1, padding=1, is_final=False):
        super(DecoderBlock_SC, self).__init__()
        if not is_final:  # use bias
            final_layer = nn.ConvTranspose2d(mid_channels, out_channels, kernel_size=2, stride=2)
            self.final_name = "trans_conv"
        else:  # whether to be the Output-Block(with 1*1conv)
            final_layer = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.final_name = "conv11"

        self.main = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size,
                                stride=strides, padding=padding, bias=False)),
            ("bn1", nn.BatchNorm2d(mid_channels)),
            ("conv2", nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size,
                                stride=strides, padding=padding, bias=False)),
            ("bn2", nn.BatchNorm2d(mid_channels)),
            (self.final_name, final_layer)
        ]))

    def forward(self, x):
        out = x
        for name, module in self.main.named_children():
            out = module(out)
            if "bn" in name:  # Add Relu after bn layer
                out = F.relu(out)

        return out


class SCBlock(nn.Module):
    """Multi-task: see as part of Unet, just initialize in Unet"""

    def __init__(self, mid_channels=256):
        # Input: batch_size*256*15*15
        super(SCBlock, self).__init__()
        self.SCBlock = nn.Sequential(OrderedDict([
            ("GlobAvgPool", nn.AdaptiveAvgPool2d((1, 1))),  # batch_size*256*1*1
            ("flatten", nn.Flatten()),  # batch_size*256
            ("linear1", nn.Linear(256, mid_channels)),  # batch_size*(15*15*256)
            ("linear2", nn.Linear(mid_channels, 1))  # bacth_size*1
        ]))

    def forward(self, x):
        out = self.SCBlock.flatten(self.SCBlock.GlobAvgPool(x))  # torch.Size([32, 256])
        out = F.relu(self.SCBlock.linear1(out))  # torch.Size([32, 256])
        out = self.SCBlock.linear2(out)  # torch.Size([32, 1])

        return out


class UNet(nn.Module):
    def __init__(self, enb_in=[1, 16, 32, 64, 128], enb_out=[16, 32, 64, 128, 256], use_SC=False,
                 deb_in=[256, 128, 64, 32], deb_mid=[128, 64, 32, 16], deb_out=[64, 32, 16, 4], init_weights=True):
        super(UNet, self).__init__()
        """Encoder"""
        # enb_list = []
        # for i in range(4):
        #     enb_list.append(EncoderBlock(enb_in[i], enb_out[i]))
        # enb_list.append(EncoderBlock(enb_in[4], enb_out[4], is_final=True))
        # self.Encoder = nn.Sequential(*enb_list)
        enb_order = OrderedDict([])
        for i in range(4):
            enb_order["enb" + str(i + 1)] = EncoderBlock_SC(enb_in[i], enb_out[i])
        enb_order["enb5"] = EncoderBlock_SC(enb_in[4], enb_out[4], is_final=True)
        self.Encoder = nn.Sequential(enb_order)
        """Decoder"""
        # deb_list = []
        # for i in range(3):
        #     deb_list.append(DecoderBlock(deb_in[i], deb_mid[i], deb_out[i]))
        # deb_list.append(DecoderBlock(deb_in[3], deb_mid[3], deb_out[3], is_final=True))
        # self.Decoder = nn.Sequential(*deb_list)
        deb_order = OrderedDict([])
        for i in range(3):
            deb_order["deb" + str(i + 1)] = DecoderBlock_SC(deb_in[i], deb_mid[i], deb_out[i])
        deb_order["deb4"] = DecoderBlock_SC(deb_in[3], deb_mid[3], deb_out[3], is_final=True)
        self.Decoder = nn.Sequential(deb_order)
        """SCNN Part"""
        if use_SC:
            self.SCBlock = nn.Sequential(OrderedDict([
                ("SCBlock", SCBlock())
            ]))
        self.use_SC = use_SC

        """Weight Init"""
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        """Encoder Part"""
        out, save1 = self.Encoder.enb1(x)
        out, save2 = self.Encoder.enb2(out)
        out, save3 = self.Encoder.enb3(out)
        out, save4 = self.Encoder.enb4(out)
        out, sc_out = self.Encoder.enb5(out)
        """Decoder Part"""
        out = self.Decoder.deb1(torch.cat([save4, out], 1))  # concat on channel-dim
        out = self.Decoder.deb2(torch.cat([save3, out], 1))
        out = self.Decoder.deb3(torch.cat([save2, out], 1))
        out = self.Decoder.deb4(torch.cat([save1, out], 1))
        """SC Part"""
        if self.use_SC:
            sc_out = self.SCBlock.SCBlock(sc_out) #torch.Size([32, 1])
            return out, torch.sigmoid(sc_out)  # Optional output: sc_out, only used in SC regualar term.
        else:
            return out #torch.Size([32, 4, 240, 240])

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.ConvTranspose2d):
                para_tuple = m.weight.data.shape
                m.weight.data.copy_(bilinear_kernel(in_channels=para_tuple[0],
                                                    out_channels=para_tuple[1], kernel_size=para_tuple[2]))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



"""SRNN Part"""
# Input: batch_size*1*w*h   Int(0,1,2,3):as regular term; (Float-mean)/str:Train para
# Output: batch_size*type_num*w*h
class EncoderBlock_SR(nn.Module):
    """2 3*3conv"""
    def __init__(self, in_channels, out_channels, mid_channels=1, kernel_size=3,
                 strides=1, padding=1, is_final=False):
        super(EncoderBlock_SR, self).__init__()
        if not is_final:
            self.main = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size,
                                   stride=2, padding=padding, bias=False)),
                ('bn1', nn.BatchNorm2d(out_channels)),
                ('conv2', nn.Conv2d(out_channels, out_channels,kernel_size=kernel_size,
                                   stride=strides, padding=padding, bias=False)),
                ('bn2', nn.BatchNorm2d(out_channels))
            ]))
        else:
            self.main = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size,
                                    stride=3, padding=padding, bias=False)),
                ('bn1', nn.BatchNorm2d(mid_channels)),
                ("Flatten", nn.Flatten()), #1*10*10 to 100
                ("linear", nn.Linear(1*10*10, out_channels))
            ]))

    def forward(self, x):
        out = x
        for name, module in self.main.named_children():
            out = module(out)
            # print(name, module, out.shape)
            # if "bn" in name or "linear" in name: #Add Relu after bn and linear layer
            if "bn" in name:
                out = F.relu(out)

        return out


class DecoderBlock_SR(nn.Module):
    """2 3*3conv + 1 2*2up-conv or 1*1conv"""
    def __init__(self, in_channels, out_channels, kernel_size=4,
                 strides=2, padding=1, is_first=False, is_final=False):
        super(DecoderBlock_SR, self).__init__()
        self.is_first = is_first
        if is_first: #Only linear and reshape layer
            self.main = nn.Sequential(OrderedDict([
                ("linear", nn.Linear(in_channels, out_channels))
            ]))

        elif is_final: #not bn and relu last time, change channel at conv
            self.main = nn.Sequential(OrderedDict([
                ("trans_conv", nn.ConvTranspose2d(in_channels, in_channels, bias=False,
                                   padding=1, kernel_size=4, stride=2)),
                ("bn1", nn.BatchNorm2d(in_channels)),
                ("conv", nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   stride=1, padding=1, bias=False))
            ]))

        else:
            self.main = nn.Sequential(OrderedDict([
                ("trans_conv", nn.ConvTranspose2d(in_channels, out_channels, bias=False,
                                   padding=padding, kernel_size=kernel_size, stride=strides)),
                ("bn1", nn.BatchNorm2d(out_channels)),
                ("conv", nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                   stride=1, padding=1, bias=False)),
                ("bn2", nn.BatchNorm2d(out_channels))
            ]))

    def forward(self, x):
        out = x
        batch_size = x.shape[0]
        for name, module in self.main.named_children():
            out = module(out)
            if "bn" in name:  # Add Relu after bn layer
                out = F.relu(out)
            if name=="linear":
                out = F.relu(out)
                out = out.reshape(batch_size,1,10,10)
            # print(name, module, out.shape)

        return out


class SRNN(nn.Module):
    """
    Input: batch_size*type_num*w*h: binary;     Loss function: cross_entropy
    Train:
        Input: Images: Gold Standard: binary; batch_size*type_num*w*h
               Labels: Gold Standard: [0,1,2,3]; batch_size*w*h
    Use as loss-term:
        Input1: Images prediction output from Unet: binary; batch_size*type_num*w*h
        Input2: Gold Standard: binary; batch_size*type_num*w*h
    """
    def __init__(self,enb_in=[4,16,32,64], enb_out=[16,32,64,64],
                 deb_in=[64,1,64,32,16], deb_out=[100,64,32,16,4], init_weights=True):
        super(SRNN, self).__init__()
        # self.FirstLayer = nn.Sequential(OrderedDict([
        #     ("conv", nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, bias=False)),
        #     ("bn", nn.BatchNorm2d(4)),
        #     ]))
        """Encoder"""
        # enb_list = []
        # for i in range(4):
        #     enb_list.append(EncoderBlock(enb_in[i], enb_out[i]))
        # enb_list.append(EncoderBlock(enb_in[4], enb_out[4], is_final=True))
        # self.Encoder = nn.Sequential(*enb_list)
        enb_order = OrderedDict([])
        for i in range(3):
            enb_order["enb"+str(i+1)] = EncoderBlock_SR(enb_in[i], enb_out[i])
        enb_order["enb4"] = EncoderBlock_SR(enb_in[3], enb_out[3], is_final=True)
        self.Encoder = nn.Sequential(enb_order)
        """Decoder"""
        # deb_list = []
        # for i in range(3):
        #     deb_list.append(DecoderBlock(deb_in[i], deb_mid[i], deb_out[i]))
        # deb_list.append(DecoderBlock(deb_in[3], deb_mid[3], deb_out[3], is_final=True))
        # self.Decoder = nn.Sequential(*deb_list)
        deb_order = OrderedDict([])
        deb_order["deb_first"] = DecoderBlock_SR(deb_in[0], deb_out[0], is_first=True)
        deb_order["deb1"] = DecoderBlock_SR(deb_in[1], deb_out[1], padding=2,
                                               kernel_size=7, strides=3)
        deb_order["deb2"] = DecoderBlock_SR(deb_in[2], deb_out[2])
        deb_order["deb3"] = DecoderBlock_SR(deb_in[3], deb_out[3])
        deb_order["deb_final"] = DecoderBlock_SR(deb_in[4], deb_out[4], is_final=True)
        self.Decoder = nn.Sequential(deb_order)

        """Weight Init"""
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # out = F.relu(self.FirstLayer(x))
        """Encoder Part"""
        out = self.Encoder.enb1(x)
        out = self.Encoder.enb2(out)
        out = self.Encoder.enb3(out)
        out = self.Encoder.enb4(out)
        # """Decoder Part"""
        out = self.Decoder.deb_first(out)
        out = self.Decoder.deb1(out)
        out = self.Decoder.deb2(out)
        out = self.Decoder.deb3(out)
        out = self.Decoder.deb_final(out) #batch_size*type_num*w*h

        return out


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


"""UNet Part(Can't deal with SC regular-term)"""
class EncoderBlock(nn.Module):
    """2 3*3conv + 1 2*2Maxpool or 2*2up-conv"""
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1, padding=1,
                 is_final=False):
        super(EncoderBlock, self).__init__()
        if not is_final: #whether to be the Transfer-Block(with no Maxpool)
            final_layer = nn.MaxPool2d(2,2)
            self.final_name = "Maxpool"
        else: #use bias
            # self.W_final = bilinear_kernel(out_channels, in_channels, kernel_size=2)
            final_layer = nn.ConvTranspose2d(out_channels, in_channels, kernel_size=2, stride=2)
            self.final_name = "trans_conv"
            # self.final_layer.weight.data.copy_(self.W_final)  # bilinear_kernel Init

        self.main = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size,
                               stride=strides, padding=padding, bias=False)),
            ('bn1', nn.BatchNorm2d(out_channels)),
            ('conv2', nn.Conv2d(out_channels, out_channels,kernel_size=kernel_size,
                               stride=strides, padding=padding, bias=False)),
            ('bn2', nn.BatchNorm2d(out_channels)),
            (self.final_name, final_layer)
        ]))

    """
    # def forward(self, x):
    # # Able to Deal with SC, since SC need the saveout of trans_conv
    #     out = x
    #     for name, module in self.main.named_children():
    #         out = module(out)
    #         # print(name, module, out.shape)
    #         if "bn" in name: #Add Relu after bn layer
    #             out = F.relu(out)
    #         if name=="bn2":
    #             out_save = out
    # 
    #     return out, out_save
    """

    def forward(self, x):
    # Deal with no_SC version, since SC need the saveout of trans_conv
        out = x
        for name, module in self.main.named_children():
            out = module(out)
            # print(name, module, out.shape)
            if "bn" in name: #Add Relu after bn layer
                out = F.relu(out)
            if name=="bn2" and self.final_name!="trans_conv":
                out_save = out

        if self.final_name!="trans_conv":
            return out, out_save
        else:
            return out


class DecoderBlock(nn.Module):
    """2 3*3conv + 1 2*2up-conv or 1*1conv"""
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size=3, strides=1, padding=1, is_final=False):
        super(DecoderBlock, self).__init__()
        if not is_final: #use bias
            final_layer = nn.ConvTranspose2d(mid_channels, out_channels, kernel_size=2, stride=2)
            self.final_name = "trans_conv"
        else:#whether to be the Output-Block(with 1*1conv)
            final_layer = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.final_name = "conv11"

        self.main = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size,
                               stride=strides, padding=padding, bias=False)),
            ("bn1", nn.BatchNorm2d(mid_channels)),
            ("conv2", nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size,
                               stride=strides, padding=padding, bias=False)),
            ("bn2", nn.BatchNorm2d(mid_channels)),
            (self.final_name, final_layer)
        ]))

    def forward(self, x):
        out = x
        for name, module in self.main.named_children():
            out = module(out)
            if "bn" in name:  # Add Relu after bn layer
                out = F.relu(out)

        return out


class UNetPro(nn.Module):
    def __init__(self,enb_in=[1,16,32,64,128], enb_out=[16,32,64,128,256], use_SC=False,
                 deb_in=[256,128,64,32], deb_mid=[128,64,32,16], deb_out=[64,32,16,4], init_weights=True):
        super(UNetPro, self).__init__()
        """Encoder"""
        # enb_list = []
        # for i in range(4):
        #     enb_list.append(EncoderBlock(enb_in[i], enb_out[i]))
        # enb_list.append(EncoderBlock(enb_in[4], enb_out[4], is_final=True))
        # self.Encoder = nn.Sequential(*enb_list)
        enb_order = OrderedDict([])
        for i in range(4):
            enb_order["enb"+str(i+1)] = EncoderBlock(enb_in[i], enb_out[i])
        enb_order["enb5"] = EncoderBlock(enb_in[4], enb_out[4], is_final=True)
        self.Encoder = nn.Sequential(enb_order)
        """Decoder"""
        # deb_list = []
        # for i in range(3):
        #     deb_list.append(DecoderBlock(deb_in[i], deb_mid[i], deb_out[i]))
        # deb_list.append(DecoderBlock(deb_in[3], deb_mid[3], deb_out[3], is_final=True))
        # self.Decoder = nn.Sequential(*deb_list)
        deb_order = OrderedDict([])
        for i in range(3):
            deb_order["deb" + str(i + 1)] = DecoderBlock(deb_in[i], deb_mid[i], deb_out[i])
        deb_order["deb4"] = DecoderBlock(deb_in[3], deb_mid[3], deb_out[3], is_final=True)
        self.Decoder = nn.Sequential(deb_order)
        """SCNN Part"""
        # self.SCBlock = nn.Sequential(OrderedDict([
        #     ("SCBlock", SCNN())
        #     ]))
        # self.use_SC = use_SC

        """Weight Init"""
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        """Encoder Part"""
        out, save1 = self.Encoder.enb1(x)
        out, save2 = self.Encoder.enb2(out)
        out, save3 = self.Encoder.enb3(out)
        out, save4 = self.Encoder.enb4(out)
        out = self.Encoder.enb5(out)
        """Decoder Part"""
        out = self.Decoder.deb1(torch.cat([save4, out], 1)) #concat on channel-dim
        out = self.Decoder.deb2(torch.cat([save3, out], 1))
        out = self.Decoder.deb3(torch.cat([save2, out], 1))
        out = self.Decoder.deb4(torch.cat([save1, out], 1))

        # if self.use_SC:
        #     return out, sc_out #Optional output: sc_out, only used in SC regualar term.
        # else:
        return out


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.ConvTranspose2d):
                para_tuple = m.weight.data.shape
                m.weight.data.copy_(bilinear_kernel(in_channels=para_tuple[0],
                                    out_channels=para_tuple[1], kernel_size=para_tuple[2]))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


"""
#test UNet
model = UNet()
inp = torch.rand(10, 1, 224, 224)
outp = model(inp)
print(outp.shape)#torch.Size([10, 4, 224, 224])
"""


class FCNNet(nn.Module):
    """FCN"""
    def __init__(self):
        super(FCNNet, self).__init__()
        self.num_classes = 4
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        pretrained_net_paras = torchvision.models.resnet18(pretrained=True)  # 使用ResNet的预训练参数，不能设为self，否则自动加入网络
        self.resnet = nn.Sequential(*list(pretrained_net_paras.children())[1:-2])
        self.final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.transpose_conv = nn.ConvTranspose2d(self.num_classes, self.num_classes, #上采样
                                    kernel_size=64, padding=16, stride=32)
        self.W = bilinear_kernel(self.num_classes, self.num_classes, 64)

    def forward(self, X):
        # 全卷积网络[用双线性插值的上采样初始化转置卷积层。对于 1×1 卷积层，我们使用Xavier初始化参数。
        self.transpose_conv.weight.data.copy_(self.W)
        return self.transpose_conv(self.final_conv(self.resnet(self.conv1(X))))