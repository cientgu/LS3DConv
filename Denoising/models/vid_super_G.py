import numpy as np
import math
import torch
import torch.nn.functional as F
import os
import sys
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from .base_model import BaseModel
from . import networks

class vid_super_G(BaseModel):
    def name(self):
        return 'vid_super_G'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain        
        if not opt.debug:
            torch.backends.cudnn.benchmark = True       
        self.input_channel = opt.label_nc
        if_sn = not opt.no_SN
        if opt.no_D == True:
            if_sn = False

        layer_num = 3
        if opt.use_yuv == False:
            input_channel = 3
        else:
            input_channel = 1
        self.netG = networks.define_G(conv_type=opt.G_conv_type, image_size=opt.loadSize, layer_num=layer_num, model_type=opt.model, input_channel=input_channel, if_sn=if_sn, padding_mode=opt.padding_mode, gpu_ids=opt.gpu_ids)

        print('---------- modelG Networks initialized -------------') 
        print('-----------------------------------------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:                    
            self.load_network(self.netG, 'G', opt.which_epoch, opt.load_pretrain)
            
        # set loss functions and optimizers
        if self.isTrain:            
            
            # initialize optimizer G
            params = list(self.netG.parameters())
            beta1, beta2 = opt.beta1, 0.999
            lr = opt.lr
            self.old_lr = opt.lr

            # self.optimizer_G = torch.optim.Adam(params, lr=lr, betas=(beta1, beta2))
            self.optimizer_G = torch.optim.Adam(params, lr=lr, betas=(beta1, beta2))

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


    def forward(self, input_A, encode_input=False):
        if encode_input == True:
            input_A = self.encode_input(input_A) 
        fake_B = self.netG(input_A)
        return fake_B


    def inference(self, input_A, encode_input=False, save_mem=True):
        if self.opt.debug == True:
            self.netG.cuda(self.opt.gpu_ids[0])
            self.netG.eval()
            if encode_input == True:
                input_A = self.encode_input(input_A) 
            temp = self.netG(input_A)
        else:
            with torch.no_grad():
                self.netG.eval()
                if encode_input == True:
                    input_A = self.encode_input(input_A) 
                if save_mem == True:
                    temp = self.netG(input_A, save_mem=True)
                    # temp = self.netG.downsample_model(input_A)
                    # print("downsample_model over ")
                    # torch.cuda.empty_cache()
                    # temp = self.netG.resblock(temp)
                    # print("resblock over ")
                    # torch.cuda.empty_cache()
                    # temp = self.netG.upsample_model(temp)
                    # print("upsample_model over ")
                    # torch.cuda.empty_cache()
                else:
                    temp = self.netG(input_A)
        return temp

    def save(self, label):        
        self.save_network(self.netG, 'G', label, self.gpu_ids)       

    def update_learning_rate(self, epoch):  
        if self.opt.no_SN == True:
            lr = self.opt.lr * (1 - (epoch - self.opt.niter) / self.opt.niter_decay)
        else:
            lr = self.opt.g_lr * (1 - (epoch - self.opt.niter) / self.opt.niter_decay)
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def update_fixed_params(self): # finetune all scales instead of just finest scale
        params = []
        for s in range(self.n_scales):
            params += list(getattr(self, 'netG'+str(s)).parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.old_lr, betas=(self.opt.beta1, 0.999))
        self.finetune_all = True
        print('------------ Now finetuning all scales -----------')
