#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

# from time import time
import time
import torch
import torch.nn as nn
from torch.autograd import gradcheck

# from modules.deform_conv import DeformConv, _DeformConv, DeformConvPack
from modules.modulated_deform_conv import ModulatedDeformConv, _ModulatedDeformConv, ModulatedDeformConvPack
# from modules.deform_psroi_pooling import DeformRoIPooling, _DeformRoIPooling, DeformRoIPoolingPack
from modules.deform_conv3d import DeformConv3d, _DeformConv3d, DeformConv3dPack

deformable_groups = 1
N, inC, inT, inH, inW = 2, 4, 4, 4, 4
outC = 4
kT, kH, kW = 3, 3, 3

torch.manual_seed(77)

def example_ten_dconv3d():
    input = torch.randn(1, 64, 14, 128, 128).cuda()
    # wrap all things (offset and mask) in DCN
    start1 = time.time()
    dcn = DeformConv3dPack(64, 64, kernel_size=(3, 3, 3), stride=(1,1,1),
              padding=(1,1,1), deformable_groups=1).cuda()
    dcn2 = DeformConv3dPack(64, 64, kernel_size=(3, 3, 3), stride=(1,1,1),
              padding=(1,1,1), deformable_groups=1).cuda()
    # dcn3 = DeformConv3dPack(64, 64, kernel_size=(3, 3, 3), stride=(1,1,1),
    #           padding=(1,1,1), deformable_groups=1).cuda()
    # dcn4 = DeformConv3dPack(64, 64, kernel_size=(3, 3, 3), stride=(1,1,1),
    #           padding=(1,1,1), deformable_groups=1).cuda()
    # dcn5 = DeformConv3dPack(64, 64, kernel_size=(3, 3, 3), stride=(1,1,1),
    #           padding=(1,1,1), deformable_groups=1).cuda()
    # dcn6 = DeformConv3dPack(64, 64, kernel_size=(3, 3, 3), stride=(1,1,1),
    #           padding=(1,1,1), deformable_groups=1).cuda()
    # dcn7 = DeformConv3dPack(64, 64, kernel_size=(3, 3, 3), stride=(1,1,1),
    #           padding=(1,1,1), deformable_groups=1).cuda()
    # dcn8 = DeformConv3dPack(64, 64, kernel_size=(3, 3, 3), stride=(1,1,1),
    #           padding=(1,1,1), deformable_groups=1).cuda()
    # dcn9 = DeformConv3dPack(64, 64, kernel_size=(3, 3, 3), stride=(1,1,1),
    #           padding=(1,1,1), deformable_groups=1).cuda()
    # dcn10 = DeformConv3dPack(64, 64, kernel_size=(3, 3, 3), stride=(1,1,1),
    #           padding=(1,1,1), deformable_groups=1).cuda()
    # dcn11 = DeformConv3dPack(64, 64, kernel_size=(3, 3, 3), stride=(1,1,1),
    #           padding=(1,1,1), deformable_groups=1).cuda()
    stop1 = time.time()
    torch.cuda.synchronize()
    print("time 1 is "+str(stop1-start1))
    start2 = time.time()
    output1 = dcn(input)
    output2 = dcn2(output1)
    # output3 = dcn3(output2)
    # output4 = dcn4(output3)
    # output5 = dcn5(output4)
    # output6 = dcn6(output5)
    # output7 = dcn7(output6)
    # output8 = dcn8(output7)
    # output9 = dcn9(output8)
    # output10 = dcn10(output9)
    # output11 = dcn11(output10)
    

    torch.cuda.synchronize()
    

    stop2 = time.time()
    print("time 2 is "+str(stop2-start2))
    start3 = time.time()
    targert = output2.new(*output2.size())
    targert.data.uniform_(-0.01, 0.01)
    error = (targert - output2).mean()
    error.backward()
    torch.cuda.synchronize()
    stop3 = time.time()
    print("time 3 is "+str(stop3-start3))
    



    # output = dcn(input)
    # targert = output.new(*output.size())
    # targert.data.uniform_(-0.01, 0.01)
    # error = (targert - output).mean()
    # error.backward()
    print(output2.shape)


def example_dconv3d():
    input = torch.randn(1, 64, 14, 128, 128).cuda()
    # wrap all things (offset and mask) in DCN
    dcn = DeformConv3dPack(64, 64, kernel_size=(3, 3, 3), stride=(1,1,1),
              padding=(1,1,1), deformable_groups=1).cuda()

    output = dcn(input)
    targert = output.new(*output.size())
    targert.data.uniform_(-0.01, 0.01)
    error = (targert - output).mean()
    error.backward()
    print(output.shape)

def conv_identify(weight, bias, groups=1):
    weight.data.zero_()
    bias.data.zero_()
    o, i, t, h, w = weight.shape
    y = h//2
    x = w//2
    k = t//2
    oc = o // groups
    for p in range(i):
        for q in range(o):
            if (p) == (q % oc):
                # print(q, p, y, x)
                # print(q % oc)
                weight.data[q, p, k, y, x] = 1.0

def check_dconv3d_zero_offset():
    conv_offset = nn.Conv3d(inC, deformable_groups * 2 * kT * kH * kW,
                            kernel_size=(kT, kH, kW),
                            stride=(1, 1, 1),
                            padding=(1, 1, 1),
                            bias=True).cuda()

    conv_mask = nn.Conv3d(inC, deformable_groups * 1 * kT * kH * kW,
                          kernel_size=(kT, kH, kW),
                          stride=(1, 1, 1),
                          padding=(1, 1, 1),
                          bias=True).cuda()

    dcn = DeformConv3d(inC, outC, (kT, kH, kW),
                   stride=(1,1,1), padding=(1,1,1), dilation=(1,1,1),
                   groups=2, 
                   deformable_groups=deformable_groups, im2col_step=1).cuda()
    pcn = nn.Conv3d(inC, outC, (kT, kH, kW), stride=(1,1,1), padding=(1,1,1), dilation=(1,1,1), groups=2).cuda()
    pcn.weight = dcn.weight
    pcn.bias = dcn.bias
    print((pcn.weight.data - dcn.weight.data).abs().max())

    conv_offset.weight.data.zero_()
    conv_offset.bias.data.zero_()
    conv_mask.weight.data.zero_()
    conv_mask.bias.data.zero_()

    input = torch.randn(N, inC, inT, inH, inW).cuda()
    offset = conv_offset(input)
    mask = conv_mask(input)
    mask = torch.sigmoid(mask)
    mask *= 2
    output_d = dcn(input, offset, mask)
    output_p = pcn(input)
    d = (output_d - output_p).abs().max()
    if d < 1e-5:
        print('dconv3d zero offset passed with {}'.format(d))
    else:
        print('dconv3d zero offset failed with {}'.format(d))
        # print(output_p)
        # print(output_d)
        print((output_d - output_p).abs())

def check_dconv3d_zero_offset_identify():
    with torch.no_grad():
        conv_offset = nn.Conv3d(inC, deformable_groups * 2 * kT * kH * kW,
                                kernel_size=(kT, kH, kW),
                                stride=(1, 1, 1),
                                padding=(1, 1, 1),
                                dilation=(1, 1, 1),
                                bias=True).cuda()

        conv_mask = nn.Conv3d(inC, deformable_groups * 1 * kT * kH * kW,
                              kernel_size=(kT, kH, kW),
                              stride=(1, 1, 1),
                              padding=(1, 1, 1),
                              dilation=(1, 1, 1),
                              bias=True).cuda()

        groups = 2
        dcn = DeformConv3d(inC, outC, (kT, kH, kW), 
            stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), 
            groups=groups, 
            deformable_groups=deformable_groups,
            im2col_step=1).cuda()

        conv_offset.weight.data.zero_()
        conv_offset.bias.data.zero_()
        conv_mask.weight.data.zero_()
        conv_mask.bias.data.zero_()
        conv_identify(dcn.weight, dcn.bias, groups)

        input = torch.randn(N, inC, inT, inH, inW).cuda()
        offset = conv_offset(input)
        mask = conv_mask(input)
        mask = torch.sigmoid(mask)
        output = dcn(input, offset, mask)
        output *= 2
        d = (input - output).abs().max()
        if d < 1e-10:
            print('dconv3d zero offset identify passed with {}'.format(d))
        else:
            print('dconv3d zero offset identify failed with {}'.format(d))
            # print(input)
            # print(output)
            print((input - output).abs())

def check_dconv3d_im2col_step_forward():
    conv_offset = nn.Conv3d(inC, deformable_groups * 2 * kT * kH * kW,
                            kernel_size=(kT, kH, kW),
                            stride=(1, 1, 1),
                            padding=(1, 1, 1),
                            bias=True).cuda()

    conv_mask = nn.Conv3d(inC, deformable_groups * 1 * kT * kH * kW,
                          kernel_size=(kT, kH, kW),
                          stride=(1, 1, 1),
                          padding=(1, 1, 1),
                          bias=True).cuda()

    input = torch.randn(N, inC, inT, inH, inW).cuda()
    offset = conv_offset(input)
    mask = conv_mask(input)
    groups = 2

    dcn1 = DeformConv3d(inC, outC, (kT, kH, kW), 
        stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), 
        groups=groups, 
        deformable_groups=deformable_groups,
        im2col_step=1).cuda()

    dcn2 = DeformConv3d(inC, outC, (kT, kH, kW), 
        stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), 
        groups=groups, 
        deformable_groups=deformable_groups,
        im2col_step=2).cuda()
    dcn1.weight = dcn2.weight
    dcn1.bias = dcn2.bias
    output1 = dcn1(input, offset, mask)
    output2 = dcn2(input, offset, mask)

    d = (output1 - output2).abs().max()
    if d < 1e-10:
        print('dconv3d im2col_step forward passed with {}'.format(d))
    else:
        print('dconv3d im2col_step forward failed with {}'.format(d))
        print(output1)
        print(output2)
        print((output1 - output2).abs())

def check_dconv3d_im2col_step_backward():
    stride = 1
    padding = 1
    groups = 2
    dilation = 1

    input = torch.rand(N, inC, inT, inH, inW).cuda() * 0.01
    input.requires_grad = True

    offset = torch.randn(N, deformable_groups * 2 * kT * kW * kH, inT, inH, inW).cuda() * 2
    # offset.data.zero_()
    # offset.data -= 0.5
    offset.requires_grad = True

    mask = torch.sigmoid(torch.randn(N, deformable_groups * 1 * kW * kH * kT, inT, inH, inW).cuda())
    mask.requires_grad = True

    weight = torch.randn(outC, int(inC//groups), kT, kH, kW).cuda()
    weight.requires_grad = True

    bias = torch.rand(outC).cuda()
    bias.requires_grad = True

    output1 = _DeformConv3d(input, offset, mask, weight, bias, stride, padding, dilation, groups, deformable_groups, 2)
    targert = torch.rand(*output1.size()).cuda()
    error = (targert - output1).mean()
    error.backward(retain_graph=True)
    input_grad = input.grad.clone()
    offset_grad = offset.grad.clone()
    mask_grad = mask.grad.clone()
    weight_grad = weight.grad.clone()
    bias_grad = bias.grad.clone()
    output2 = _DeformConv3d(input, offset, mask, weight, bias, stride, padding, dilation, groups, deformable_groups, 1)
    error2 = (targert - output2).mean()
    error.backward()
    print((output1 - output2).abs().max())
    input_grad_err = (input.grad - 2 * input_grad).abs().max() 
    offset_grad_err = (offset.grad - 2 * offset_grad).abs().max()
    mask_grad_err = (mask.grad - 2 * mask_grad).abs().max()
    weight_grad_err = (weight.grad - 2 * weight_grad).abs().max()
    bias_grad_err = (bias.grad - 2 * bias_grad).abs().max()
    grad_err = input_grad_err + offset_grad_err + mask_grad_err + weight_grad_err + bias_grad_err
    if grad_err < 1e-7:
        print("dconv3d im2col_step backward passed with {}".format(grad_err))
    else:
        print("dconv3d im2col_step backward failed with {}".format(grad_err))


def check_gradient_dconv3d():
    stride = 1
    padding = 0
    groups = 1
    dilation = 1
    im2col_step = 1

    # inT = 1
    # kT = 1
    # kH = 3
    # kW = 3
    inT = 4
    inH = 5
    inW = 5

    print(kT, kH, kW)

    input = torch.rand(N, inC, inT, inH, inW).cuda() * 1 # 0.01
    # input = torch.zeros(N, inC, inT, inH, inW).cuda() * 0.01
    input.requires_grad = True

    offset = torch.randn(N, deformable_groups * 2 * kT * kW * kH, inT, inH, inW).cuda() * 2
    # offset = torch.zeros(N, deformable_groups * 2 * kT * kW * kH, inT, inH, inW).cuda() * 2
    # offset.data.zero_()
    # offset.data -= 0.5
    offset.requires_grad = True

    mask = torch.rand(N, deformable_groups * 1 * kT * kW * kH, inT, inH, inW).cuda()
    # mask = torch.zeros(N, deformable_groups * 1 * kT * kW * kH, inT, inH, inW).cuda()
    # mask.data.zero_()
    mask.requires_grad = True
    mask = torch.sigmoid(mask)

    weight = torch.randn(outC, int(inC//groups), kT, kH, kW).cuda()
    # weight = torch.zeros(outC, int(inC//groups), kT, kH, kW).cuda()
    weight.requires_grad = True

    bias = torch.rand(outC).cuda()
    # bias = torch.zeros(outC).cuda()
    bias.requires_grad = True

    # input = input.double()
    # offset = offset.double()
    # mask = mask.double()
    # weight = weight.double()
    # bias = bias.double()

    print('check_gradient_dconv3d: ',
          gradcheck(_DeformConv3d, (input, offset, mask, weight, bias,
                    stride, padding, dilation, groups, deformable_groups, im2col_step),
                    eps=1e-3, atol=1e-3, rtol=1e-2, raise_exception=True))

    # print('check_gradient_dconv3d: ',
    #       gradcheck(_DeformConv3d, (input, offset, mask, weight, bias,
    #                 stride, padding, dilation, groups, deformable_groups, im2col_step)))

if __name__ == '__main__':

#    example_ten_dconv3d()
    # example_dconv3d()

    # print('checking')
    # check_dconv3d_im2col_step_forward()
    # print("1 finished  ---------------")
    # check_dconv3d_im2col_step_backward()
    # print("2 finished  ---------------")
    # if inC == outC:
    #     check_dconv3d_zero_offset()
    #     print("3 finished  ------------")
    #     check_dconv3d_zero_offset_identify()
    #     print("4 finished  ------------")

    torch.manual_seed(49)
    check_gradient_dconv3d()
    print("5 finished  ------------")
