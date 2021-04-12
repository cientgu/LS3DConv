#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import math
from torch import nn
from torch.nn import init
from torch.nn.modules.utils import _pair,_triple

from functions.deform_conv3d_func import DeformConv3dFunction

class DeformConv3d(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dilation=1, groups=1, deformable_groups=1, im2col_step=64, bias=True):
        super(DeformConv3d, self).__init__()

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
        self.deformable_groups = deformable_groups
        self.im2col_step = im2col_step
        self.use_bias = bias

        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels//groups, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()
        if not self.use_bias:
            self.bias.requires_grad = False

    def reset_parameters(self):
        n = self.in_channels
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input, offset, mask):
        print(" did i go throuth here ???????????????????????????????????????????????????????? ")
        assert 2 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]== \
            offset.shape[1]
        assert self.deformable_groups * self.kernel_size[0] * self.kernel_size[1]  * self.kernel_size[2]== \
            mask.shape[1]
        return DeformConv3dFunction.apply(input, offset, mask,
                                                   self.weight,
                                                   self.bias,
                                                   self.stride,
                                                   self.padding,
                                                   self.dilation,
                                                   self.groups,
                                                   self.deformable_groups,
                                                   self.im2col_step)

_DeformConv3d = DeformConv3dFunction.apply

class DeformConv3dPack(DeformConv3d):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, groups=1, deformable_groups=1, im2col_step=64, bias=True, lr_mult=0.1):
        super(DeformConv3dPack, self).__init__(in_channels, out_channels,
                                  kernel_size, stride, padding, dilation, groups, deformable_groups, im2col_step, bias)

        out_channels = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        # probably kernel_size should be (self.kernel_size[0]-1)

        self.conv_offset_mask = nn.Conv3d(self.in_channels,
                                          out_channels,
                                          kernel_size=self.kernel_size,
                                          stride=self.stride,
                                          padding=self.padding,
                                          bias=True)
        self.conv_offset_mask.lr_mult = lr_mult
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)     # split on channel dim
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        mask = mask*2
        return DeformConv3dFunction.apply(input, offset, mask, 
                                                self.weight, 
                                                self.bias, 
                                                self.stride, 
                                                self.padding, 
                                                self.dilation, 
                                                self.groups,
                                                self.deformable_groups,
                                                self.im2col_step)



class DeformConv3dPack_v2(DeformConv3d):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, groups=1, deformable_groups=1, im2col_step=64, bias=True, lr_mult=0.1):
        super(DeformConv3dPack_v2, self).__init__(in_channels, out_channels,
                                  kernel_size, stride, padding, dilation, groups, deformable_groups, im2col_step, bias)

        out_channels = self.deformable_groups * 3 * self.kernel_size[0] # * self.kernel_size[1] * self.kernel_size[2]
        # probably kernel_size should be (self.kernel_size[0]-1)

        self.conv_offset_mask = nn.Conv3d(self.in_channels,
                                          out_channels,
                                          kernel_size=self.kernel_size,
                                          stride=self.stride,
                                          padding=self.padding,
                                          bias=True)
        self.conv_offset_mask.lr_mult = lr_mult
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        out_size = out.size()
        spatial_kernel = self.kernel_size[1] * self.kernel_size[2]
        assert out.dim() == 5
        out_offset = out.unsqueeze(2).repeat(1,1,spatial_kernel,1,1,1).reshape(out_size[0],out_size[1]*spatial_kernel,out_size[2],out_size[3],out_size[4])

        o1, o2, mask = torch.chunk(out_offset, 3, dim=1)     # split on channel dim
        offset = torch.cat((o1, o2), dim=1)
        
        mask = torch.sigmoid(mask)
        mask = mask*2
        return DeformConv3dFunction.apply(input, offset, mask, 
                                                self.weight, 
                                                self.bias, 
                                                self.stride, 
                                                self.padding, 
                                                self.dilation, 
                                                self.groups,
                                                self.deformable_groups,
                                                self.im2col_step)
