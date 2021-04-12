import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out

class Res3DBlock(nn.Module):
    def __init__(self, planes, stride=1, downsample=None):
        super(Res3DBlock, self).__init__()


        pad1 = nn.ReplicationPad3d((1,1,1,1,1,1))
        conv1 = nn.Conv3d(planes,planes,kernel_size=(3,3,3),stride=(1, 1, 1),padding=(0, 0, 0),bias=False)
        bn1 = nn.BatchNorm3d(planes)
        relu = nn.ReLU(inplace=True)
        pad2 = nn.ReplicationPad3d((1,1,1,1,1,1))
        conv2 = nn.Conv3d(planes,planes,kernel_size=(3,3,3),stride=(1, 1, 1),padding=(0, 0, 0),bias=False)
        bn2 = nn.BatchNorm3d(planes)

        model = [pad1,conv1,bn1,relu,pad2,conv2,bn2]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = x + self.model(x)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

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


class ResNet(nn.Module):

    def __init__(self,block,layers,sample_size,sample_duration,shortcut_type='B',num_classes=400,input_channel=35):
        # BasicBlock
        # [3,4,6,3] for res34
        # [2,2,2,2] for res18


        self.inplanes = 64
        self.input_channel = 35
        super(ResNet, self).__init__()

        pad1_1 = nn.ReplicationPad3d((3,3,3,3,1,1))
        conv1_1 = nn.Conv3d(
            self.input_channel,
            64,
            kernel_size=(3,7,7),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            bias=False)
        bn1_1 = nn.BatchNorm3d(64)
        relu = nn.ReLU(inplace=True)

        conv1_2 = nn.Conv3d(
            64,
            128,
            kernel_size=(3,3,3),
            stride=(2, 2, 2),
            padding=(1, 1, 1),
            bias=False)
        bn1_2 = nn.BatchNorm3d(128)
        relu = nn.ReLU(inplace=True)

        downsample_model = [pad1_1,conv1_1,bn1_1,relu,conv1_2,bn1_2,relu]
        self.downsample_model = nn.Sequential(*downsample_model)

        resblock = []
        resblock.append(Res3DBlock(128))
        resblock.append(Res3DBlock(128))
        resblock.append(Res3DBlock(128))
        resblock.append(Res3DBlock(128))

        self.resblock = nn.Sequential(*resblock)

        deconv2_1 = nn.ConvTranspose3d(128,64,kernel_size=(4,4,4),stride=(2,2,2),padding=(1,1,1),bias=False)
        bn2_1 = nn.BatchNorm3d(64)
        relu = nn.ReLU(inplace=True)
        conv2_2 = nn.Conv3d(64,64,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1),bias=False)
        pad2_2 = nn.ReplicationPad3d((3,3,3,3,1,1))
        conv2_3 = nn.Conv3d(64,3,kernel_size=(3,7,7),stride=(1,1,1),padding=(0,0,0),bias=False)
        tanh = nn.Tanh()


  #   (10): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
  #   (11): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #   (12): ReLU(inplace)
  #   (13): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
  #   (14): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #   (15): ReLU(inplace)
  # )
  # (bg_encoder): Sequential(
  #   (0): ReflectionPad2d((3, 3, 3, 3))
  #   (1): Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1))
  #   (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  #   (3): ReLU(inplace)
  # )
  # (bg_decoder): Sequential(
  #   (0): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
  #   (1): ReflectionPad2d((3, 3, 3, 3))
  #   (2): Conv2d(64, 3, kernel_size=(7, 7), stride=(1, 1))
  #   (3): Tanh()


        upsample_model = [deconv2_1, bn2_1, relu, conv2_2, pad2_2, conv2_3, tanh]
        self.upsample_model = nn.Sequential(*upsample_model)


        # last_duration = int(math.ceil(sample_duration / 16))
        # last_size = int(math.ceil(sample_size / 32))
        # self.avgpool = nn.AvgPool3d(
        #     (last_duration, last_size, last_size), stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def encode_input(self, input_map):
        size = input_map.size()
        self.bs, input_channel, tG, self.height, self.width = size[0], size[1], size[2], size[3], size[4]
        
        input_map = input_map.data.cuda()                ### can be better to save memory
        if self.input_channel != 0:                        
            # create one-hot vector for label map             
            oneHot_size = (self.bs, self.input_channel, tG, self.height, self.width)
            input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, input_map.long(), 1.0)    
            input_map = input_label        
        input_map = Variable(input_map)
        return input_map

    def forward(self, x, encode_input=True):
        if encode_input == True:
            x = self.encode_input(x)            # [1, 35, 4, 128, 256]
        # x = self.conv1(x)

        # print("2 X size is --- ")               # [1, 64, 4, 64, 128]
        # print(x.size())

        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        # print("5 X size is --- ")               # [1, 64, 2, 32, 64]
        # print(x.size())

        # x = self.layer1(x)
        # x = self.layer2(x)
        # print("7 X size is --- ")               # [1, 128, 1, 16, 32]
        # print(x.size())
        # x = self.layer3(x)
        # print("8 X size is --- ")               # [1, 256, 1, 8, 16]
        # print(x.size())
        # x = self.layer4(x)
        # print("9 X size is --- ")               # [1, 512, 1, 4, 8]
        # print(x.size())
        # x = self.avgpool(x)
        # print("10 X size is --- ")              # [1, 512, 1, 1, 5]
        # print(x.size())
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        x = self.downsample_model(x)
        x = self.resblock(x)
        x = self.upsample_model(x)
        return x


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


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(Res3DBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model
