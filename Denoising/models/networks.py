import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

from torch.nn import init
import functools
import numpy as np
import copy
import torch.nn.utils.spectral_norm as spectral_norm

from modules.deform_conv3d import DeformConv3d, _DeformConv3d, DeformConv3dPack, DeformConv3dPack_v2
from torch.nn.modules.utils import _pair,_triple

class Res3DBlock(nn.Module):
    def __init__(self, planes, if_bias=False, if_sn=False, padding_mode="const", if_relu=False, conv_type='deform', addition_back_pad=False):
        super(Res3DBlock, self).__init__()
        # self.relu = nn.ReLU(inplace=True)
        self.if_relu = if_relu
        self.conv_type = conv_type
        self.addition_back_pad = addition_back_pad

        pad1 = nn.ReplicationPad3d((1,1,1,1,1,1))
        if self.conv_type == "deformv2":
            conv1 = DeformConv3dPack_v2(planes,planes,kernel_size=(3,3,3),stride=(1, 1, 1),padding=(0, 0, 0),bias=if_bias)
        elif self.conv_type == "deform":
            conv1 = DeformConv3dPack(planes,planes,kernel_size=(3,3,3),stride=(1, 1, 1),padding=(0, 0, 0),bias=if_bias)
        elif self.conv_type == "shiftv1":
            conv1 = ShiftConv3d_v1(planes,planes,kernel_size=(1,3,3),stride=(1,1,1),padding=(0,0,0),bias=if_bias,if_sn=if_sn)
        elif self.conv_type == "shiftv3":
            conv1 = ShiftConv3d_v3(planes,planes,kernel_size=(3,3,3),stride=(1,1,1),padding=(0,0,0),bias=if_bias,if_sn=if_sn)
        elif self.conv_type == "normal":
            conv1 = nn.Conv3d(planes,planes,kernel_size=(3,3,3),stride=(1, 1, 1),padding=(0, 0, 0),bias=if_bias)
        else:
            print("not this conv_type ")

        bn1 = nn.BatchNorm3d(planes)
        relu = nn.ReLU(inplace=True)
        pad2 = nn.ReplicationPad3d((1,1,1,1,1,1))
        conv2 = nn.Conv3d(planes,planes,kernel_size=(3,3,3),stride=(1, 1, 1),padding=(0, 0, 0),bias=if_bias)
        bn2 = nn.BatchNorm3d(planes)

        if if_sn == True:
            conv2 = spectral_norm(conv2)
            if self.conv_type != "shiftv1" and self.conv_type != "shiftv3":
                conv1 = spectral_norm(conv1)


        if padding_mode == "const":
            pad1 = nn.ReplicationPad3d((1,1,1,1,1,1))
            pad2 = nn.ReplicationPad3d((1,1,1,1,1,1))
            model = [pad1,conv1,bn1,relu,pad2,conv2,bn2]
        elif padding_mode == "zero":
            pad1 = nn.ConstantPad3d((1,1,1,1,1,1),0)
            pad2 = nn.ConstantPad3d((1,1,1,1,1,1),0)
            model = [pad1,conv1,bn1,relu,pad2,conv2,bn2]
        elif padding_mode == "shuyang":
            pad1 = nn.ReplicationPad3d((1,1,1,1,0,0))
            pad2 = nn.ReplicationPad3d((1,1,1,1,0,0))
            expand1 = FeatExpand(5)
            expand2 = FeatExpand(5)
            model = [pad1,conv1,expand1,bn1,relu,pad2,conv2,expand2,bn2]
        else:
            print("NotImplementedError in padding_mode")

        if self.addition_back_pad == True:
            self.pad3 = nn.ReplicationPad3d((0,0,0,0,0,1))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        if self.addition_back_pad == True:
            out = self.pad3(out)
        if self.if_relu == True:
            relu = nn.ReLU(inplace=True)
            return relu(out)
        else:
            return out

class DeconvUpBlock(nn.Module):
    def __init__(self, planes, if_sn=False, if_relu=False):
        super(DeconvUpBlock, self).__init__()

        self.if_relu = if_relu

        deconv = nn.ConvTranspose3d(planes,planes,kernel_size=(3,3,3),stride=(2,1,1),padding=(1,1,1))        
        bn = nn.BatchNorm3d(planes)
        if if_sn == True:
            deconv = spectral_norm(deconv)

        model = [deconv,bn]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        if self.if_relu == True:
            relu = nn.ReLU(inplace=True)
            return relu(out)
        else:
            return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class FeatExpand(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1):
        super(FeatExpand, self).__init__()
        self.dim = dim
        assert kernel_size == 3
        assert stride == 1

    def forward(self, x):
        assert x.dim() == self.dim
        (batch,channel,frames,height,width) = x.size()
        pad1 = nn.ConstantPad3d((0,0,0,0,1,1),0)
        pad2 = nn.ConstantPad3d((0,0,0,0,2,0),0)
        pad3 = nn.ConstantPad3d((0,0,0,0,0,2),0)

        sum_tensor = pad1(x) + pad2(x) + pad3(x)
        out = sum_tensor.clone()
        frames = sum_tensor.size()[2]
        for index in range(frames):
            if index == 0 or index == frames-1:
                out[:,:,index,:,:] = sum_tensor[:,:,index,:,:]/1
            elif index == 1 or index == frames-2:
                out[:,:,index,:,:] = sum_tensor[:,:,index,:,:]/2
            else:
                out[:,:,index,:,:] = sum_tensor[:,:,index,:,:]/3

        return out


class ResNet(nn.Module):

    def __init__(self,conv_type,image_size,block,layer_num,input_channel=35, if_sn=False, padding_mode="const"):
        if_bias = True

        self.conv_type = conv_type
        self.padding_mode = padding_mode
        self.inplanes = 64
        self.input_channel = input_channel
        super(ResNet, self).__init__()


        conv1_1 = nn.Conv3d(
            self.input_channel,
            64,
            kernel_size=(3,7,7),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            bias=if_bias)
        bn1_1 = nn.BatchNorm3d(64)
        relu = nn.ReLU(inplace=True)

        conv1_2 = nn.Conv3d(
            64,
            128,
            kernel_size=(3,3,3),
            stride=(1, 2, 2),
            padding=(0, 0, 0),
            bias=if_bias)
        bn1_2 = nn.BatchNorm3d(128)
        relu = nn.ReLU(inplace=True)

        if image_size == 256:
            conv1_3 = nn.Conv3d(
            128,
            256,
            kernel_size=(3,3,3),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            bias=if_bias)
        elif image_size == 512:
            conv1_3 = nn.Conv3d(
            128,
            256,
            kernel_size=(3,3,3),
            stride=(1, 2, 2),
            padding=(0, 0, 0),
            bias=if_bias)
        else:
            print("NotImplementedError")

        bn1_3 = nn.BatchNorm3d(256)
        relu = nn.ReLU(inplace=True)

        if if_sn == True:
            conv1_1 = spectral_norm(conv1_1)
            conv1_2 = spectral_norm(conv1_2)
            conv1_3 = spectral_norm(conv1_3)


        if self.padding_mode == "const":
            pad1_1 = nn.ReplicationPad3d((3,3,3,3,1,1))
            pad1_2 = nn.ReplicationPad3d((1,1,1,1,1,1))
            pad1_3 = nn.ReplicationPad3d((1,1,1,1,1,1))
            downsample_model = [pad1_1,conv1_1,bn1_1,relu,pad1_2,conv1_2,bn1_2,relu,pad1_3,conv1_3,bn1_3,relu]
        elif self.padding_mode == "zero":
            pad1_1 = nn.ConstantPad3d((3,3,3,3,1,1),0)
            pad1_2 = nn.ConstantPad3d((1,1,1,1,1,1),0)
            pad1_3 = nn.ConstantPad3d((1,1,1,1,1,1),0)
            downsample_model = [pad1_1,conv1_1,bn1_1,relu,pad1_2,conv1_2,bn1_2,relu,pad1_3,conv1_3,bn1_3,relu]
        elif self.padding_mode == "shuyang":
            pad1_1 = nn.ReplicationPad3d((3,3,3,3,0,0))
            pad1_2 = nn.ReplicationPad3d((1,1,1,1,0,0))
            pad1_3 = nn.ReplicationPad3d((1,1,1,1,0,0))
            expand1_1 = FeatExpand(5)
            expand1_2 = FeatExpand(5)
            expand1_3 = FeatExpand(5)
            downsample_model = [pad1_1,conv1_1,expand1_1,bn1_1,relu,pad1_2,conv1_2,expand1_2,bn1_2,relu,pad1_3,conv1_3,expand1_3,bn1_3,relu]
        else:
            print("NotImplementedError in padding_mode")

        self.downsample_model = nn.Sequential(*downsample_model)

        resblock = []
        for layer_index in range(layer_num):
            resblock.append(Res3DBlock(256,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode, if_relu=False, conv_type=conv_type))
        
        # resblock.append(Res3DBlock(256,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode))
        # resblock.append(Res3DBlock(256,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode))
        # resblock.append(Res3DBlock(256,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode))
        # resblock.append(Res3DBlock(256,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode))
        # resblock.append(Res3DBlock(256,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode))
        # resblock.append(Res3DBlock(256,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode))
        # resblock.append(Res3DBlock(256,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode))
        # resblock.append(Res3DBlock(256,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode))
        # resblock.append(Res3DBlock(256,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode))

        self.resblock = nn.Sequential(*resblock)

        deconv2_1 = nn.ConvTranspose3d(256,128,kernel_size=(3,4,4),stride=(1,2,2),padding=(1,1,1),bias=if_bias)
        bn2_1 = nn.BatchNorm3d(128)
        relu = nn.ReLU(inplace=True)
        if image_size == 256:        
            deconv2_2 = nn.ConvTranspose3d(128,128,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1),bias=if_bias)
        elif image_size == 512:
            deconv2_2 = nn.ConvTranspose3d(128,128,kernel_size=(3,4,4),stride=(1,2,2),padding=(1,1,1),bias=if_bias)
        else:
            print("NotImplementedError")
        bn2_2 = nn.BatchNorm3d(128)
        relu = nn.ReLU(inplace=True)

        conv2_2 = nn.Conv3d(128,64,kernel_size=(3,3,3),stride=(1,1,1),padding=(0,0,0),bias=if_bias)
        bn2_3 = nn.BatchNorm3d(64)
        if self.padding_mode == "shuyang":
            conv2_3 = nn.Conv3d(64,3,kernel_size=(1,7,7),stride=(1,1,1),padding=(0,0,0),bias=if_bias)
        else:
            # conv2_3 = nn.Conv3d(64,3,kernel_size=(3,7,7),stride=(1,1,1),padding=(0,0,0),bias=if_bias)
            conv2_3 = nn.Conv3d(64,3,kernel_size=(7,7,7),stride=(1,1,1),padding=(0,0,0),bias=if_bias)
        tanh = nn.Tanh()

        if if_sn == True:
            deconv2_1 = spectral_norm(deconv2_1)
            deconv2_2 = spectral_norm(deconv2_2)
            conv2_2 = spectral_norm(conv2_2)
            # conv2_3 = spectral_norm(conv2_3)

        if self.padding_mode == "const":
            pad2_2 = nn.ReplicationPad3d((1,1,1,1,1,1))
            # pad2_3 = nn.ReplicationPad3d((3,3,3,3,1,1))
            pad2_3 = nn.ReplicationPad3d((3,3,3,3,3,3))
            upsample_model = [deconv2_1,bn2_1,relu,deconv2_2,bn2_2,relu,pad2_2,conv2_2,bn2_3,pad2_3,conv2_3,tanh]
        elif self.padding_mode == "zero":
            pad2_2 = nn.ConstantPad3d((1,1,1,1,1,1),0)
            # pad2_3 = nn.ConstantPad3d((3,3,3,3,1,1),0)
            pad2_3 = nn.ConstantPad3d((3,3,3,3,3,3),0)
            upsample_model = [deconv2_1,bn2_1,relu,deconv2_2,bn2_2,relu,pad2_2,conv2_2,bn2_3,pad2_3,conv2_3,tanh]
        elif self.padding_mode == "shuyang":
            pad2_2 = nn.ReplicationPad3d((1,1,1,1,0,0))
            pad2_3 = nn.ReplicationPad3d((3,3,3,3,0,0))
            expand2_2 = FeatExpand(5)
            upsample_model = [deconv2_1,bn2_1,relu,deconv2_2,bn2_2,relu,pad2_2,conv2_2,expand2_2,bn2_3,pad2_3,conv2_3,tanh]
        else:
            print("NotImplementedError in padding_mode")        

        self.upsample_model = nn.Sequential(*upsample_model)

    def forward(self, x, encode_input=True):
        x = self.downsample_model(x)
        x = self.resblock(x)
        x = self.upsample_model(x)
        return x


class P3DBlock(nn.Module):
    def __init__(self, planes, if_bias=False, if_sn=False, padding_mode="const", conv_type='deform', addition_back_pad=False):
        super(P3DBlock, self).__init__()
        
        self.conv_type = conv_type
        
        prelu = nn.PReLU()

        conv1 = nn.Conv3d(planes,planes,kernel_size=(1,3,3),stride=(1,1,1),padding=(0,1,1),bias=if_bias)

        if self.conv_type == "normal":
            conv2 = nn.Conv3d(planes,planes,kernel_size=(3,1,1),stride=(1,1,1),padding=(1,0,0),bias=if_bias)
        elif self.conv_type == "deform_v2":
            conv2 = DeformConv3dPack_v2(planes,planes,kernel_size=(3,1,1),stride=(1,1,1),padding=(1,0,0),bias=if_bias)
        else:
            print("NotImplementedError")

        model = [prelu,conv1,conv2]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = x + self.model(x)
        return out

class LSRBlock(nn.Module):
    def __init__(self, planes, if_bias=False, if_sn=False, padding_mode="const", conv_type='deform', addition_back_pad=False):
        super(LSRBlock, self).__init__()
        
        self.conv_type = conv_type
        


        if self.conv_type == "normal":
            conv1 = nn.Conv3d(planes,planes*4,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1),bias=if_bias)
        elif self.conv_type == "deform_v2":
            conv1 = DeformConv3dPack_v2(planes,planes*4,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1),bias=if_bias)
        else:
            print("NotImplementedError")

        upsample = nn.ConvTranspose3d(planes*4,planes//4,kernel_size=(3,3,3),stride=(1,4,4),padding=(1,0,0),output_padding=(0,1,1),bias=if_bias)
        

        if self.conv_type == "normal":
            conv2 = nn.Conv3d(planes//4,1,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1),bias=if_bias)
        elif self.conv_type == "deform_v2":
            conv2 = DeformConv3dPack_v2(planes//4,1,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1),bias=if_bias)
        else:
            print("NotImplementedError")


        relu = nn.LeakyReLU(0.2, True)
        model = [conv1,upsample,conv2]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        return out


class SuperNet_b(nn.Module):
    def __init__(self,conv_type,image_size,block,layer_num,input_channel=3, if_sn=False, padding_mode="const"):
        super(SuperNet_b, self).__init__()
        if_bias = True
        self.conv_type = conv_type
        self.padding_mode = "const"
        self.inplanes = 64
        self.input_channel = 1
        

        if self.conv_type == "normal":
            self.conv1 = nn.Conv3d(self.input_channel,64,kernel_size=(3,3,3),stride=(1, 1, 1),padding=(1, 1, 1),bias=if_bias)
        elif self.conv_type == "deform_v2":
            self.conv1 = DeformConv3dPack_v2(self.input_channel,64,kernel_size=(3,3,3),stride=(1, 1, 1),padding=(1, 1, 1),bias=if_bias)
        else:
            print("NotImplementedError")

        self.relu = nn.LeakyReLU(0.2, True)

        lr_block = []
        for layer_index in range(5):
            lr_block.append(P3DBlock(64,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode, conv_type=conv_type))
        
        self.lr_block = nn.Sequential(*lr_block)

        lsr_block = []
        prelu = nn.PReLU()
        dropout = nn.Dropout(p=0.3)
        lsr_block.append(prelu)
        lsr_block.append(dropout)
        lsr_block.append(LSRBlock(64,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode, conv_type=conv_type))
        # lsr_block.append(LSRBlock(64,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode, conv_type=conv_type))
        self.lsr_block = nn.Sequential(*lsr_block)


    def forward(self, x, save_mem=False):
        b,c,t,h,w = x.size()
        out = self.conv1(x)
        out = self.relu(out)
        out = out + self.lr_block(out)
        out = self.lsr_block(out)
        upsample_tensor = F.interpolate(x.reshape(b*c,t,h,w),scale_factor=4,mode='bilinear').reshape(b,c,t,4*h,4*w)
        out = out + upsample_tensor
        return out

class SuperNet(nn.Module):
    def __init__(self,conv_type,image_size,block,layer_num,input_channel=3, if_sn=False, padding_mode="const"):
        super(SuperNet, self).__init__()
        if_bias = True
        self.conv_type = conv_type
        self.padding_mode = "const"
        self.inplanes = 64
        self.input_channel = input_channel
        

        self.conv1 = nn.Conv3d(3,64,kernel_size=(3,3,3),stride=(1, 1, 1),padding=(1, 1, 1),bias=if_bias)

        layer1 = []
        layer2 = []
        layer3 = []

        deconv1 = nn.ConvTranspose3d(64,64,kernel_size=(3,4,4),stride=(1,2,2),padding=(1,1,1),bias=if_bias)
        deconv2 = nn.ConvTranspose3d(64,64,kernel_size=(3,4,4),stride=(1,2,2),padding=(1,1,1),bias=if_bias)

        layer1.append(Res3DBlock(64,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode, if_relu=False, conv_type=conv_type))
        layer1.append(Res3DBlock(64,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode, if_relu=False, conv_type=conv_type))
        layer1.append(deconv1)
        self.layer1 = nn.Sequential(*layer1)
        layer2.append(Res3DBlock(64,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode, if_relu=False, conv_type="normal"))
        layer2.append(Res3DBlock(64,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode, if_relu=False, conv_type="normal"))
        layer2.append(deconv2)
        self.layer2 = nn.Sequential(*layer2)
        layer3.append(Res3DBlock(64,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode, if_relu=False, conv_type="normal"))
        # layer3.append(Res3DBlock(64,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode, if_relu=False, conv_type=conv_type))
        layer3.append(nn.Conv3d(64,input_channel,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1),bias=if_bias))
        self.layer3 = nn.Sequential(*layer3)

    def forward(self, x, save_mem=False):
        b,c,t,h,w = x.size()
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        if self.input_channel == 1:
            upsample_tensor = F.interpolate(x[:,0:1,:,:,:].reshape(b*1,t,h,w),scale_factor=4,mode='bilinear').reshape(b,1,t,4*h,4*w)
        else:
            upsample_tensor = F.interpolate(x.reshape(b*c,t,h,w),scale_factor=4,mode='bilinear').reshape(b,c,t,4*h,4*w)
        out = out + upsample_tensor
        return out

class DenoiseNet(nn.Module):
    def __init__(self,conv_type,image_size,block,layer_num,input_channel=3, if_sn=False, padding_mode="const"):
        super(DenoiseNet, self).__init__()
        if_bias = True
        self.conv_type = conv_type
        self.padding_mode = "const"
        self.input_channel = input_channel
        
        pad1_1 = nn.ReplicationPad3d((1,1,1,1,1,1))
        conv1_1 = nn.Conv3d(3,64,kernel_size=(3,3,3),stride=(1, 2, 2),padding=(0, 0, 0),bias=if_bias)
        bn1_1 = nn.BatchNorm3d(64)
        relu = nn.ReLU(inplace=True)

        pad1_2 = nn.ReplicationPad3d((1,1,1,1,1,1))
        conv1_2 = nn.Conv3d(64,128,kernel_size=(3,3,3),stride=(1, 2, 2),padding=(0, 0, 0),bias=if_bias)
        bn1_2 = nn.BatchNorm3d(128)
        relu = nn.ReLU(inplace=True)

        downsample_model = [pad1_1,conv1_1,bn1_1,relu,pad1_2,conv1_2,bn1_2,relu]
        self.downsample_model = nn.Sequential(*downsample_model)

        layer1 = []
        # layer1.append(Res3DBlock(64,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode, if_relu=False, conv_type=conv_type))
        # layer1.append(Res3DBlock(128,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode, if_relu=False, conv_type=conv_type))
        layer1.append(Res3DBlock(128,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode, if_relu=False, conv_type=conv_type))
        layer1.append(Res3DBlock(128,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode, if_relu=False, conv_type=conv_type))
        self.layer1 = nn.Sequential(*layer1)

        layer2 = []
        layer2.append(nn.ConvTranspose3d(128,128,kernel_size=(3,4,4),stride=(1,2,2),padding=(1,1,1),bias=if_bias))
        layer2.append(nn.BatchNorm3d(128))
        layer2.append(nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(*layer2)
        
        layer3 = []
        layer3.append(nn.ConvTranspose3d(128,64,kernel_size=(3,4,4),stride=(1,2,2),padding=(1,1,1),bias=if_bias))
        layer3.append(nn.BatchNorm3d(64))
        layer3.append(nn.ReLU(inplace=True))
        layer3.append(nn.Conv3d(64,3,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1),bias=if_bias))
        self.layer3 = nn.Sequential(*layer3)


    def forward(self, x, save_mem=False):
        b,c,t,h,w = x.size()
        out = self.downsample_model(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = out + x
        return out

class InterNet(nn.Module):

    def __init__(self,conv_type,image_size,block,layer_num,input_channel=3, if_sn=False, padding_mode="const"):
        if_bias = True

        self.conv_type = conv_type
        self.padding_mode = "const"
        self.inplanes = 64
        self.input_channel = 3
        super(InterNet, self).__init__()


        conv1_1 = nn.Conv3d(
            self.input_channel,
            64,
            kernel_size=(3,7,7),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            bias=if_bias)
        bn1_1 = nn.BatchNorm3d(64)
        relu = nn.ReLU(inplace=True)

        conv1_2 = nn.Conv3d(
            64,
            128,
            kernel_size=(3,3,3),
            stride=(1, 2, 2),
            padding=(0, 0, 0),
            bias=if_bias)
        bn1_2 = nn.BatchNorm3d(128)
        relu = nn.ReLU(inplace=True)


        conv1_3 = nn.Conv3d(
        128,
        256,
        kernel_size=(3,3,3),
        stride=(1, 2, 2),
        padding=(0, 0, 0),
        bias=if_bias)


        bn1_3 = nn.BatchNorm3d(256)
        relu = nn.ReLU(inplace=True)

        if if_sn == True:
            conv1_1 = spectral_norm(conv1_1)
            conv1_2 = spectral_norm(conv1_2)
            conv1_3 = spectral_norm(conv1_3)


        if self.padding_mode == "const":
            pad1_1 = nn.ReplicationPad3d((3,3,3,3,1,1))
            pad1_2 = nn.ReplicationPad3d((1,1,1,1,1,1))
            pad1_3 = nn.ReplicationPad3d((1,1,1,1,1,1))
            downsample_model = [pad1_1,conv1_1,bn1_1,relu,pad1_2,conv1_2,bn1_2,relu,pad1_3,conv1_3,bn1_3,relu]
        else:
            print("NotImplementedError in padding_mode")

        self.downsample_model = nn.Sequential(*downsample_model)

        resblock = []

        deconv1 = DeconvUpBlock(256, if_sn=if_sn, if_relu=False)
        deconv2 = DeconvUpBlock(256, if_sn=if_sn, if_relu=False)
        deconv3 = DeconvUpBlock(256, if_sn=if_sn, if_relu=False)

        if layer_num >= 3:
            resblock.append(deconv1)
        else:
            resblock.append(Res3DBlock(256,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode, if_relu=False, conv_type=conv_type))    
        resblock.append(Res3DBlock(256,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode, if_relu=False, conv_type=conv_type))
        resblock.append(Res3DBlock(256,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode, if_relu=False, conv_type=conv_type))
        if layer_num >= 2:
            resblock.append(deconv2)
        else:
            resblock.append(Res3DBlock(256,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode, if_relu=False, conv_type=conv_type))    
        resblock.append(Res3DBlock(256,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode, if_relu=False, conv_type=conv_type))
        resblock.append(Res3DBlock(256,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode, if_relu=False, conv_type=conv_type))
        if layer_num >= 1:
            resblock.append(deconv3)
        else:
            resblock.append(Res3DBlock(256,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode, if_relu=False, conv_type=conv_type))    
        resblock.append(Res3DBlock(256,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode, if_relu=False, conv_type=conv_type))
        resblock.append(Res3DBlock(256,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode, if_relu=False, conv_type=conv_type))

        
            






        self.resblock = nn.Sequential(*resblock)

        deconv2_1 = nn.ConvTranspose3d(256,128,kernel_size=(3,4,4),stride=(1,2,2),padding=(1,1,1),bias=if_bias)
        bn2_1 = nn.BatchNorm3d(128)
        relu = nn.ReLU(inplace=True)


        deconv2_2 = nn.ConvTranspose3d(128,128,kernel_size=(3,4,4),stride=(1,2,2),padding=(1,1,1),bias=if_bias)

        bn2_2 = nn.BatchNorm3d(128)
        relu = nn.ReLU(inplace=True)

        conv2_2 = nn.Conv3d(128,64,kernel_size=(3,3,3),stride=(1,1,1),padding=(0,0,0),bias=if_bias)
        bn2_3 = nn.BatchNorm3d(64)
        if self.padding_mode == "shuyang":
            conv2_3 = nn.Conv3d(64,3,kernel_size=(1,7,7),stride=(1,1,1),padding=(0,0,0),bias=if_bias)
        else:
            conv2_3 = nn.Conv3d(64,3,kernel_size=(3,7,7),stride=(1,1,1),padding=(0,0,0),bias=if_bias)
        tanh = nn.Tanh()

        if if_sn == True:
            deconv2_1 = spectral_norm(deconv2_1)
            deconv2_2 = spectral_norm(deconv2_2)
            conv2_2 = spectral_norm(conv2_2)
            # conv2_3 = spectral_norm(conv2_3)

        if self.padding_mode == "const":
            pad2_2 = nn.ReplicationPad3d((1,1,1,1,1,1))
            pad2_3 = nn.ReplicationPad3d((3,3,3,3,1,1))
            upsample_model = [deconv2_1,bn2_1,relu,deconv2_2,bn2_2,relu,pad2_2,conv2_2,bn2_3,pad2_3,conv2_3,tanh]
        elif self.padding_mode == "zero":
            pad2_2 = nn.ConstantPad3d((1,1,1,1,1,1),0)
            pad2_3 = nn.ConstantPad3d((3,3,3,3,1,1),0)
            upsample_model = [deconv2_1,bn2_1,relu,deconv2_2,bn2_2,relu,pad2_2,conv2_2,bn2_3,pad2_3,conv2_3,tanh]
        elif self.padding_mode == "shuyang":
            pad2_2 = nn.ReplicationPad3d((1,1,1,1,0,0))
            pad2_3 = nn.ReplicationPad3d((3,3,3,3,0,0))
            expand2_2 = FeatExpand(5)
            upsample_model = [deconv2_1,bn2_1,relu,deconv2_2,bn2_2,relu,pad2_2,conv2_2,expand2_2,bn2_3,pad2_3,conv2_3,tanh]
        else:
            print("NotImplementedError in padding_mode")        

        self.upsample_model = nn.Sequential(*upsample_model)

    def forward(self, x, save_mem=False):
        if save_mem == False:
            x = self.downsample_model(x)
            x = self.resblock(x)
            x = self.upsample_model(x)
        else:
            for layer in self.downsample_model:
                x = layer(x)
                torch.cuda.empty_cache()
            for layer in self.resblock:
                x = layer(x)            
                torch.cuda.empty_cache()
            for layer in self.upsample_model:
                x = layer(x)
                torch.cuda.empty_cache()
        return x

class PredNet(nn.Module):

    def __init__(self,conv_type,image_size,block,layer_num,input_channel=3, if_sn=False, padding_mode="const"):
        if_bias = True

        self.conv_type = conv_type
        self.padding_mode = "const"
        self.inplanes = 64
        self.input_channel = 3
        super(PredNet, self).__init__()


        conv1_1 = nn.Conv3d(
            self.input_channel,
            64,
            kernel_size=(3,7,7),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            bias=if_bias)
        bn1_1 = nn.BatchNorm3d(64)
        relu = nn.ReLU(inplace=True)

        conv1_2 = nn.Conv3d(
            64,
            128,
            kernel_size=(3,3,3),
            stride=(1, 2, 2),
            padding=(0, 0, 0),
            bias=if_bias)
        bn1_2 = nn.BatchNorm3d(128)
        relu = nn.ReLU(inplace=True)

        if image_size == 256:
            conv1_3 = nn.Conv3d(
            128,
            256,
            kernel_size=(3,3,3),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            bias=if_bias)
        elif image_size == 512:
            conv1_3 = nn.Conv3d(
            128,
            256,
            kernel_size=(3,3,3),
            stride=(1, 2, 2),
            padding=(0, 0, 0),
            bias=if_bias)
        else:
            print("NotImplementedError")

        bn1_3 = nn.BatchNorm3d(256)
        relu = nn.ReLU(inplace=True)

        if if_sn == True:
            conv1_1 = spectral_norm(conv1_1)
            conv1_2 = spectral_norm(conv1_2)
            conv1_3 = spectral_norm(conv1_3)


        if self.padding_mode == "const":
            pad1_1 = nn.ReplicationPad3d((3,3,3,3,1,1))
            pad1_2 = nn.ReplicationPad3d((1,1,1,1,1,1))
            pad1_3 = nn.ReplicationPad3d((1,1,1,1,1,1))
            downsample_model = [pad1_1,conv1_1,bn1_1,relu,pad1_2,conv1_2,bn1_2,relu,pad1_3,conv1_3,bn1_3,relu]
        else:
            print("NotImplementedError in padding_mode")

        self.downsample_model = nn.Sequential(*downsample_model)

        resblock = []

        deconv1 = DeconvUpBlock(256, if_sn=if_sn, if_relu=False)
        deconv2 = DeconvUpBlock(256, if_sn=if_sn, if_relu=False)
        deconv3 = DeconvUpBlock(256, if_sn=if_sn, if_relu=False)

        if layer_num >= 3:
            resblock.append(Res3DBlock(256,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode, if_relu=False, conv_type=conv_type, addition_back_pad=True))
        else:
            resblock.append(Res3DBlock(256,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode, if_relu=False, conv_type=conv_type))    
        resblock.append(Res3DBlock(256,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode, if_relu=False, conv_type=conv_type))
        resblock.append(Res3DBlock(256,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode, if_relu=False, conv_type=conv_type))
        if layer_num >= 2:
            resblock.append(Res3DBlock(256,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode, if_relu=False, conv_type=conv_type, addition_back_pad=True))
        else:
            resblock.append(Res3DBlock(256,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode, if_relu=False, conv_type=conv_type))    
        resblock.append(Res3DBlock(256,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode, if_relu=False, conv_type=conv_type))
        resblock.append(Res3DBlock(256,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode, if_relu=False, conv_type=conv_type))
        if layer_num >= 1:
            resblock.append(Res3DBlock(256,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode, if_relu=False, conv_type=conv_type, addition_back_pad=True))
        else:
            resblock.append(Res3DBlock(256,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode, if_relu=False, conv_type=conv_type))    
        resblock.append(Res3DBlock(256,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode, if_relu=False, conv_type=conv_type))
        resblock.append(Res3DBlock(256,if_bias=if_bias, if_sn=if_sn, padding_mode=self.padding_mode, if_relu=False, conv_type=conv_type))

        
            






        self.resblock = nn.Sequential(*resblock)

        deconv2_1 = nn.ConvTranspose3d(256,128,kernel_size=(3,4,4),stride=(1,2,2),padding=(1,1,1),bias=if_bias)
        bn2_1 = nn.BatchNorm3d(128)
        relu = nn.ReLU(inplace=True)
        if image_size == 256:        
            deconv2_2 = nn.ConvTranspose3d(128,128,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1),bias=if_bias)
        elif image_size == 512:
            deconv2_2 = nn.ConvTranspose3d(128,128,kernel_size=(3,4,4),stride=(1,2,2),padding=(1,1,1),bias=if_bias)
        else:
            print("NotImplementedError")
        bn2_2 = nn.BatchNorm3d(128)
        relu = nn.ReLU(inplace=True)

        conv2_2 = nn.Conv3d(128,64,kernel_size=(3,3,3),stride=(1,1,1),padding=(0,0,0),bias=if_bias)
        bn2_3 = nn.BatchNorm3d(64)
        if self.padding_mode == "shuyang":
            conv2_3 = nn.Conv3d(64,3,kernel_size=(1,7,7),stride=(1,1,1),padding=(0,0,0),bias=if_bias)
        else:
            conv2_3 = nn.Conv3d(64,3,kernel_size=(3,7,7),stride=(1,1,1),padding=(0,0,0),bias=if_bias)
        tanh = nn.Tanh()

        if if_sn == True:
            deconv2_1 = spectral_norm(deconv2_1)
            deconv2_2 = spectral_norm(deconv2_2)
            conv2_2 = spectral_norm(conv2_2)
            # conv2_3 = spectral_norm(conv2_3)

        if self.padding_mode == "const":
            pad2_2 = nn.ReplicationPad3d((1,1,1,1,1,1))
            pad2_3 = nn.ReplicationPad3d((3,3,3,3,1,1))
            upsample_model = [deconv2_1,bn2_1,relu,deconv2_2,bn2_2,relu,pad2_2,conv2_2,bn2_3,pad2_3,conv2_3,tanh]
        elif self.padding_mode == "zero":
            pad2_2 = nn.ConstantPad3d((1,1,1,1,1,1),0)
            pad2_3 = nn.ConstantPad3d((3,3,3,3,1,1),0)
            upsample_model = [deconv2_1,bn2_1,relu,deconv2_2,bn2_2,relu,pad2_2,conv2_2,bn2_3,pad2_3,conv2_3,tanh]
        elif self.padding_mode == "shuyang":
            pad2_2 = nn.ReplicationPad3d((1,1,1,1,0,0))
            pad2_3 = nn.ReplicationPad3d((3,3,3,3,0,0))
            expand2_2 = FeatExpand(5)
            upsample_model = [deconv2_1,bn2_1,relu,deconv2_2,bn2_2,relu,pad2_2,conv2_2,expand2_2,bn2_3,pad2_3,conv2_3,tanh]
        else:
            print("NotImplementedError in padding_mode")        

        self.upsample_model = nn.Sequential(*upsample_model)

    def forward(self, x, save_mem=False):
        if save_mem == False:
            x = self.downsample_model(x)
            x = self.resblock(x)
            x = self.upsample_model(x)
        else:
            for layer in self.downsample_model:
                x = layer(x)
                torch.cuda.empty_cache()
            for layer in self.resblock:
                x = layer(x)            
                torch.cuda.empty_cache()
            for layer in self.upsample_model:
                x = layer(x)
                torch.cuda.empty_cache()
        return x


################################################


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def define_G(conv_type, image_size, layer_num, model_type, input_channel, if_sn, padding_mode, gpu_ids):
    block = Res3DBlock
    if model_type == "mask2vid":
        netG = ResNet(conv_type, image_size, block, layer_num, input_channel, if_sn, padding_mode)
    elif model_type == "vidinter":
        netG = InterNet(conv_type, image_size, block, layer_num, input_channel, if_sn, padding_mode)
    elif model_type == "vidpredict":
        netG = PredNet(conv_type, image_size, block, layer_num, input_channel, if_sn, padding_mode)
    elif model_type == "vidsuper":
        # netG = SuperNet_b(conv_type, image_size, block, layer_num, input_channel, if_sn, padding_mode)
        netG = SuperNet(conv_type, image_size, block, layer_num, input_channel, if_sn, padding_mode)
    elif model_type == "viddenoise":
        netG = DenoiseNet(conv_type, image_size, block, layer_num, input_channel, if_sn, padding_mode)
    else:
        print("something wrong -- ")


    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG



#########################################################################################
#########################################################################################

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv3d') != -1:
#         m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
#     elif classname.find('BatchNorm3d') != -1:
#         m.weight.data.fill_(1)
#         m.bias.data.zero_()
#     else:
#         print("in weights_init ")
#         print(classname)

def weights_init(m):
    classname = m.__class__.__name__
    no_param_list = ["InterNet","DeconvUpBlock","MultiscaleDiscriminator","AvgPool2d","Sequential","LeakyReLU","ReplicationPad3d","ReLU","Res3DBlock","Tanh","ResNet","AvgPool3d","AvgPool2d","MultiscaleDiscriminator_2D","FeatExpand","ConstantPad3d","PredNet"]
    if classname.find('Conv3d') != -1 and classname.find('ShiftConv3d') == -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('ConvTranspose3d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm3d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('ConvTranspose2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    else:
        if classname not in no_param_list:
            print("in weights_init ")
            print(classname)
    if classname == "DeformConv3dPack" or classname == "DeformConv3dPack_v2":
        m.conv_offset_mask.weight.data.zero_()
        m.conv_offset_mask.bias.data.zero_()

#################################################################################################


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_D(conv_type, image_size, input_nc, ndf, n_layers_D, norm, use_sigmoid, num_D, getIntermFeat, if_sn, padding_mode, gpu_ids):
    norm_layer = get_norm_layer(norm_type=norm)   
    if_bias = True
    image_size = 512
    netD = MultiscaleDiscriminator(image_size, input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat,if_bias=if_bias, if_sn=if_sn, padding=padding_mode, conv_type=conv_type)
    
    if len(gpu_ids) > 0:
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, image_size, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, 
                 use_sigmoid=False, num_D=3, getIntermFeat=False, if_bias=False, if_sn=True, padding="const", conv_type="deform"):
        super(MultiscaleDiscriminator, self).__init__()
        ## self.num_D = num_D
        self.num_D = 1
        num_D = 1
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminator(image_size, input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat, if_bias, if_sn, padding, conv_type)
            if getIntermFeat:
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool3d(kernel_size=[1,3,3], stride=[1,2,2], padding=[0,1,1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
        
                result.append(model[i](result[-1]))
            return result[-2:]
        else:
            return [model(input)]

    def forward(self, input):
        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, image_size, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, use_sigmoid=False, getIntermFeat=False, if_bias=False, if_sn=False, padding_mode="const", conv_type="deform"):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        
        self.n_layers = n_layers

        assert padding_mode == "const"

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))

        # conv1 = nn.Conv3d(input_nc,ndf,kernel_size=(3,3,3),stride=(2, 2, 2),padding=(0,0,0),bias=if_bias)
        conv1 = nn.Conv3d(input_nc,ndf,kernel_size=(3,3,3),stride=(1, 2, 2),padding=(0,0,0),bias=if_bias)
        if if_sn == True:
            conv1 = spectral_norm(conv1)

        pad1 = nn.ReplicationPad3d((1,1,1,1,1,1))
        sequence = [[pad1,conv1,nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(0, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            if conv_type == "deform":
                conv_s = DeformConv3dPack(nf_prev,nf,kernel_size=(3,3,3),stride=(2, 2, 2),padding=(0, 0, 0),bias=if_bias)
            elif conv_type == "deformv2":
                conv_s = DeformConv3dPack_v2(nf_prev,nf,kernel_size=(3,3,3),stride=(2, 2, 2),padding=(0, 0, 0),bias=if_bias)
                # conv_s = DeformConv3dPack(nf_prev,nf,kernel_size=(3,3,3),stride=(1, 2, 2),padding=(0, 0, 0),bias=if_bias)
            elif conv_type == "normal":
                conv_s = nn.Conv3d(nf_prev,nf,kernel_size=(3,3,3),stride=(2, 2, 2),padding=(0, 0, 0),bias=if_bias)
                # conv_s = nn.Conv3d(nf_prev,nf,kernel_size=(3,3,3),stride=(1, 2, 2),padding=(0, 0, 0),bias=if_bias)
            # elif conv_type == "shiftv1":
            #     conv_s = ShiftConv3d_v1(nf_prev,nf,kernel_size=(1,3,3),stride=(1,1,1),padding=(0,0,0),bias=if_bias,if_sn=if_sn)
            # elif conv_type == "shiftv3":
            #     conv_s = ShiftConv3d_v3(nf_prev,nf,kernel_size=(3,3,3),stride=(1,1,1),padding=(0,0,0),bias=if_bias,if_sn=if_sn)
            else:
                print("NotImplementedError for conv in D ")

            if if_sn == True:
                # if conv_type == "deform" or conv_type == "normal":
                conv_s = spectral_norm(conv_s)

            sequence += [[ pad1,conv_s, norm_layer(nf), nn.LeakyReLU(0.2, True) ]]

        nf_prev = nf
        nf = min(nf * 2, 512)

        pad2 = nn.ReplicationPad3d((1,1,1,1,0,0))
        conv3 = nn.Conv3d(nf,1,kernel_size=(1,3,3),stride=(1, 1, 1),padding=(0, 0, 0),bias=if_bias)

        if if_sn == True:
            conv3 = spectral_norm(conv3)

        sequence += [[pad2,conv3]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                print("debug in netD --- ")
                print(n)
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[-2:]
        else:
            return self.model(input)        


####################################################################################




def get_norm_layer_2D(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_D_2D(image_size, input_nc, ndf, n_layers_D, norm, use_sigmoid, num_D, getIntermFeat, if_sn, gpu_ids):
    norm_layer = get_norm_layer_2D(norm_type=norm)   
    if_bias = True
    image_size = 512
    netD_2D = MultiscaleDiscriminator_2D(image_size, input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat,if_bias=if_bias,if_sn=if_sn)
    
    if len(gpu_ids) > 0:
        netD_2D.cuda(gpu_ids[0])
    netD_2D.apply(weights_init)
    return netD_2D

class MultiscaleDiscriminator_2D(nn.Module):
    def __init__(self, image_size, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, use_sigmoid=False, num_D=3, getIntermFeat=False, if_bias=False, if_sn=True):
        super(MultiscaleDiscriminator_2D, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminator_2D(image_size, input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat, if_bias, if_sn)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)


    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
        
                result.append(model[i](result[-1]))
            return result[-2:]
        else:
            return [model(input)]

    def reshape_tensor(self,input):
        this_size = input.size()
        out = input.transpose(1,2).contiguous().view(-1,this_size[1],this_size[3],this_size[4])
        return out


    def forward(self, input):
        assert input.dim() == 5
        input = self.reshape_tensor(input)
        assert input.dim() == 4

        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator_2D(nn.Module):
    def __init__(self, image_size, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, use_sigmoid=False, getIntermFeat=False, if_bias=False, if_sn=False):
        super(NLayerDiscriminator_2D, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))

        conv1 = nn.Conv2d(input_nc,ndf,kernel_size=4,stride=(2, 2),padding=(1, 1),bias=if_bias)
        if if_sn == True:
            conv1 = spectral_norm(conv1)

        sequence = [[conv1, nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            conv_s = nn.Conv2d(nf_prev,nf,kernel_size=4,stride=(2, 2),padding=(1, 1),bias=if_bias)
            if if_sn == True:
                conv_s = spectral_norm(conv_s)
            sequence += [[ conv_s, norm_layer(nf), nn.LeakyReLU(0.2, True)]]

        nf_prev = nf
        nf = min(nf * 2, 512)

        if image_size == 256:
            conv2 = nn.Conv2d(nf_prev,nf,kernel_size=3,stride=(1, 1),padding=(1, 1),bias=if_bias)
        elif image_size == 512:
            conv2 = nn.Conv2d(nf_prev,nf,kernel_size=4,stride=(2, 2),padding=(1, 1),bias=if_bias)
        else:
            print("NotImplementedError")

        conv3 = nn.Conv2d(nf,1,kernel_size=3,stride=(1, 1),padding=(1, 1),bias=if_bias)

        if if_sn == True:
            conv2 = spectral_norm(conv2)
            conv3 = spectral_norm(conv3)

        sequence += [[ conv2, norm_layer(nf), nn.LeakyReLU(0.2, True)]]

        sequence += [[conv3]]


        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[-2:]
        else:
            return self.model(input)     




class ShiftConv3d_v3(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=True, if_sn=False):
        super(ShiftConv3d_v3, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels {} must be divisible by groups {}'.format(in_channels, groups))
        if out_channels % groups != 0:
            raise ValueError('out_channels {} must be divisible by groups {}'.format(out_channels, groups))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.groups = groups
        self.use_bias = bias

        assert self.kernel_size[0] == 3
        # self.padding = (0,1,1)
        self.conv3d_v1 = nn.Conv3d(in_channels,out_channels,kernel_size=(1,3,3),stride=self.stride,padding=self.padding,dilation=1,groups=1,bias=True)
        self.conv3d_v2 = nn.Conv3d(in_channels,out_channels,kernel_size=(1,3,3),stride=self.stride,padding=self.padding,dilation=1,groups=1,bias=True)
        self.conv3d_v3 = nn.Conv3d(in_channels,out_channels,kernel_size=(1,3,3),stride=self.stride,padding=self.padding,dilation=1,groups=1,bias=True)

        if if_sn == True:
            self.conv3d_v1 = spectral_norm(self.conv3d_v1)
            self.conv3d_v2 = spectral_norm(self.conv3d_v2)
            self.conv3d_v3 = spectral_norm(self.conv3d_v3)

        self.offset = self.offset_generation(out_channels)

    def offset_generation(self,channel):
        assert channel == 128 or channel == 256
        if channel == 128:
            h_list = [(-2,0,1),(-1,0,0),(-1,0,1),(-1,0,2),(1,0,0),(1,0,-1),(1,0,-2),(2,0,-1),(0,0,-1),(0,0,0),(0,0,1)]   #11
            w_list = [(-2,0,1),(-1,0,0),(-1,0,1),(-1,0,2),(1,0,0),(1,0,-1),(1,0,-2),(2,0,-1),(0,0,-1),(0,0,0),(0,0,1)]
            offset_list = []
            for i in range(11):
                for j in range(11):
                    h = h_list[i]
                    w = w_list[j]
                    offset_list.append(h+w)
            addition_offset = [(0,0,0,0,0,0)]*(128-11*11)
            offset_list += addition_offset
            assert len(offset_list) == 128
        else:
            h_list = [(-2,0,1),(-1,0,0),(-1,0,1),(-1,0,2),(1,0,0),(1,0,-1),(1,0,-2),(2,0,-1),(0,0,-1),(0,0,0),(0,0,1)]   #11
            w_list = [(-2,0,1),(-1,0,0),(-1,0,1),(-1,0,2),(1,0,0),(1,0,-1),(1,0,-2),(2,0,-1),(0,0,-1),(0,0,0),(0,0,1)]
            offset_list = []
            for i in range(11):
                for j in range(11):
                    h = h_list[i]
                    w = w_list[j]
                    offset_list.append(h+w)
                    offset_list.append(h+w)
            addition_offset = [(0,0,0,0,0,0)]*(128-11*11)*2
            offset_list += addition_offset
            # offset_list = offset_list*2
            assert len(offset_list) == 256

        return offset_list
        # offset_1d_tensor = torch.FloatTensor(offset_list)        
        # return offset_1d_tensor


    def forward(self, input):
        temp_t1 = self.conv3d_v1(input[:,:,0:-2,:,:])
        temp_t2 = self.conv3d_v2(input[:,:,1:-1,:,:])
        temp_t3 = self.conv3d_v3(input[:,:,2:,:,:])

        batch, channel, size_t, size_h, size_w = temp_t1.size()[0], temp_t1.size()[1], temp_t1.size()[2], temp_t1.size()[3], temp_t1.size()[4]
        out_tensor = torch.Tensor(batch,channel,size_t,size_h,size_w).type_as(input)
        this_pad = nn.ReplicationPad3d((2,2,2,2,0,0))
        temp_pad_t1 = this_pad(temp_t1)
        temp_pad_t2 = this_pad(temp_t2)
        temp_pad_t3 = this_pad(temp_t3)
        
        if channel == 128:
            for channel_index in range(channel):
                this_offset = self.offset[channel_index]
                frame1_tensor = temp_pad_t1[:,channel_index,:,2-this_offset[0]:2-this_offset[0]+size_h,2-this_offset[3]:2-this_offset[3]+size_w]
                frame2_tensor = temp_pad_t2[:,channel_index,:,2:2+size_h,2:2+size_w]
                frame3_tensor = temp_pad_t3[:,channel_index,:,2-this_offset[2]:2-this_offset[2]+size_h,2-this_offset[5]:2-this_offset[5]+size_w]
                this_channel_out = frame1_tensor + frame2_tensor + frame3_tensor
                out_tensor[:,channel_index,:,:,:] = this_channel_out
        elif channel == 256:
            for channel_index in range(channel):
                if channel_index%2 == 0:
                    # assert self.offset[channel_index] == self.offset[channel_index+1]
                    this_offset = self.offset[channel_index]
                    frame1_tensor = temp_pad_t1[:,channel_index:channel_index+2,:,2-this_offset[0]:2-this_offset[0]+size_h,2-this_offset[3]:2-this_offset[3]+size_w]
                    frame2_tensor = temp_pad_t2[:,channel_index:channel_index+2,:,2:2+size_h,2:2+size_w]
                    frame3_tensor = temp_pad_t3[:,channel_index:channel_index+2,:,2-this_offset[2]:2-this_offset[2]+size_h,2-this_offset[5]:2-this_offset[5]+size_w]
                    this_channel_out = frame1_tensor + frame2_tensor + frame3_tensor
                    out_tensor[:,channel_index:channel_index+2,:,:,:] = this_channel_out
        else:
            print("channel not right ")

        return out_tensor

class ShiftConv3d_v1(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=True ,if_sn=False):
        super(ShiftConv3d_v1, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels {} must be divisible by groups {}'.format(in_channels, groups))
        if out_channels % groups != 0:
            raise ValueError('out_channels {} must be divisible by groups {}'.format(out_channels, groups))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.groups = groups
        self.use_bias = bias

        assert self.kernel_size[0] == 1
        # self.padding = (0,1,1)
        self.conv3d = nn.Conv3d(in_channels,out_channels,kernel_size=self.kernel_size,stride=self.stride,padding=self.padding,dilation=1,groups=1,bias=True)
        if if_sn == True:
            self.conv3d = spectral_norm(self.conv3d)

        self.offset = self.offset_generation(out_channels)

    def offset_generation(self,channel):
        assert channel == 128 or channel == 256
        if channel == 128:
            h_list = [(-2,0,1),(-1,0,0),(-1,0,1),(-1,0,2),(1,0,0),(1,0,-1),(1,0,-2),(2,0,-1),(0,0,-1),(0,0,0),(0,0,1)]   #11
            w_list = [(-2,0,1),(-1,0,0),(-1,0,1),(-1,0,2),(1,0,0),(1,0,-1),(1,0,-2),(2,0,-1),(0,0,-1),(0,0,0),(0,0,1)]
            offset_list = []
            for i in range(11):
                for j in range(11):
                    h = h_list[i]
                    w = w_list[j]
                    offset_list.append(h+w)
            addition_offset = [(0,0,0,0,0,0)]*(128-11*11)
            offset_list += addition_offset
            assert len(offset_list) == 128
        else:
            h_list = [(-2,0,1),(-1,0,0),(-1,0,1),(-1,0,2),(1,0,0),(1,0,-1),(1,0,-2),(2,0,-1),(0,0,-1),(0,0,0),(0,0,1)]   #11
            w_list = [(-2,0,1),(-1,0,0),(-1,0,1),(-1,0,2),(1,0,0),(1,0,-1),(1,0,-2),(2,0,-1),(0,0,-1),(0,0,0),(0,0,1)]
            offset_list = []
            for i in range(11):
                for j in range(11):
                    h = h_list[i]
                    w = w_list[j]
                    offset_list.append(h+w)
                    offset_list.append(h+w)
            addition_offset = [(0,0,0,0,0,0)]*(128-11*11)*2
            offset_list += addition_offset
            # offset_list = offset_list*2
            assert len(offset_list) == 256

        return offset_list
        # offset_1d_tensor = torch.FloatTensor(offset_list)        
        # return offset_1d_tensor


    def forward(self, input):
        
        temp = self.conv3d(input)

        batch, channel, size_t, size_h, size_w = temp.size()[0], temp.size()[1], temp.size()[2], temp.size()[3], temp.size()[4]
        out_tensor = torch.Tensor(batch,channel,size_t-2,size_h,size_w).type_as(input)
        this_pad = nn.ReplicationPad3d((2,2,2,2,0,0))
        temp_pad = this_pad(temp)
        if channel == 128:
            for channel_index in range(channel):
                this_offset = self.offset[channel_index]
                frame1_tensor = temp_pad[:,channel_index,0:size_t-2,2-this_offset[0]:2-this_offset[0]+size_h,2-this_offset[3]:2-this_offset[3]+size_w]
                frame2_tensor = temp_pad[:,channel_index,1:size_t-1,2:2+size_h,2:2+size_w]
                frame3_tensor = temp_pad[:,channel_index,2:size_t,2-this_offset[2]:2-this_offset[2]+size_h,2-this_offset[5]:2-this_offset[5]+size_w]
                this_channel_out = frame1_tensor + frame2_tensor + frame3_tensor
                out_tensor[:,channel_index,:,:,:] = this_channel_out
        elif channel == 256:
            for channel_index in range(channel):
                if channel_index%2 == 0:
                    # assert self.offset[channel_index] == self.offset[channel_index+1]
                    this_offset = self.offset[channel_index]
                    frame1_tensor = temp_pad[:,channel_index:channel_index+2,0:size_t-2,2-this_offset[0]:2-this_offset[0]+size_h,2-this_offset[3]:2-this_offset[3]+size_w]
                    frame2_tensor = temp_pad[:,channel_index:channel_index+2,1:size_t-1,2:2+size_h,2:2+size_w]
                    frame3_tensor = temp_pad[:,channel_index:channel_index+2,2:size_t,2-this_offset[2]:2-this_offset[2]+size_h,2-this_offset[5]:2-this_offset[5]+size_w]
                    this_channel_out = frame1_tensor + frame2_tensor + frame3_tensor
                    out_tensor[:,channel_index:channel_index+2,:,:,:] = this_channel_out
        else:
            print("channel not right ")

        return out_tensor
