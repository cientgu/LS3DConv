### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import os
import sys
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from .base_model import BaseModel
from . import networks
from . import losses

class vid_inter_D(BaseModel):
    def name(self):
        return 'vid_inter_D'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)        
        
        if not opt.debug:
            torch.backends.cudnn.benchmark = True    

        netD_input_nc = 3
        assert opt.ndf == 64
        # assert opt.n_layers_D == 3 
        use_sigmoid = False
        # assert opt.num_D == 1
        ganFeat_loss_2d = not opt.no_ganFeat_2D
        ganFeat_loss_3d = not opt.no_ganFeat_3D

        if_sn = not opt.no_SN
        if opt.no_3D == False:
            self.netD = networks.define_D(opt.D_conv_type, opt.loadSize, netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, opt.num_D, ganFeat_loss_3d, if_sn, opt.padding_mode, opt.gpu_ids)
        self.netD_2D = networks.define_D_2D(opt.loadSize, netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, opt.num_D, ganFeat_loss_2d, if_sn, opt.gpu_ids)
        
        print('---------- modelD Networks initialized -------------') 
        print('-----------------------------------------------')

        # load networks
        if opt.continue_train or opt.load_pretrain:          
            pass
            # if opt.no_3D == False:
            #     self.load_network(self.netD, 'D', opt.which_epoch, opt.load_pretrain)            
            # self.load_network(self.netD_2D, 'D_2D', opt.which_epoch, opt.load_pretrain)                

        # set loss functions and optimizers          
        
        # define loss functions
    
        self.criterionGAN = losses.GANLoss(gan_mode = self.opt.gan_mode)
        self.criterionFeat = torch.nn.L1Loss()
        self.criterionL2 = torch.nn.MSELoss()
        self.criterionL1 = torch.nn.L1Loss() 
        self.criterionVGG = losses.VGGLoss(self.opt.gpu_ids[0], weights=None)
        self.criterionSSIM = losses.SSIM_Loss()

        self.loss_names = ["2D_fake", "2D_real", "2G_GAN", "D_fake", "D_real", "G_GAN", "loss_image", "2D_GAN_Feat", "3D_GAN_Feat", "VGG_loss"]

        # initialize optimizers D and D_T                                            
        if opt.no_3D == False:
            params_D = list(self.netD.parameters())
        params_D_2D = list(self.netD_2D.parameters())

        beta1, beta2 = opt.beta1, 0.999
        lr = opt.lr
        self.old_lr = opt.lr
        if opt.no_SN == False:
            # use TTUR
            beta1, beta2 = 0, 0.9
            lr = opt.d_lr
            self.old_lr = opt.d_lr

        # self.optimizer_D = torch.optim.Adam(params_D + params_D_2D, lr=lr, betas=(beta1, beta2))        
        if opt.no_3D == False:
            self.optimizer_D = torch.optim.Adam(params_D + params_D_2D, lr=lr, betas=(beta1, beta2))
        else:              
            self.optimizer_D = torch.optim.Adam(params_D_2D, lr=lr, betas=(beta1, beta2))

    def forward(self, fake_B, input_B):

        # change to cuda tensor
        input_B = input_B.type_as(fake_B)
        
        # GAN loss 
        #########################################
        loss_D_fake = loss_D_real = loss_G_GAN = 0    
        loss_2D_fake = loss_2D_real = loss_2G_GAN = 0    
        # Fake Detection and Loss

        # gt_in, gt_gen = self.split_tensor(input_B)
        # fa_in, fa_gen = self.split_tensor(fake_B)
        # assert fa_gen.size()[2] == 3
        # assert fa_in.size()[2] == 2
        # assert fake_B.size()[2] == 5

        # L1 loss
        loss_image = self.criterionL1(fake_B[:,:,1:2,:,:], input_B[:,:,1:2,:,:])
        loss_image += self.criterionL1(fake_B, input_B)
        loss_image = loss_image * self.opt.weight_L2
        # loss_image += (1 - self.criterionSSIM(fa_gen, gt_gen))*0.02


        zero_tensor = loss_image.clone()
        zero_tensor = zero_tensor * 0

        loss_D_fake = loss_D_real = loss_G_GAN = zero_tensor
        loss_2D_fake = loss_2D_real = loss_2G_GAN = zero_tensor
        loss_3D_GAN_Feat = zero_tensor
        loss_2D_GAN_Feat = zero_tensor
        """
        if self.opt.no_gan == True:        
            # 3D GAN loss ###########################
            if self.opt.no_3D == False:
                pred_fake_pool = self.netD.forward(fake_B.detach())
                loss_D_fake = self.criterionGAN(pred_fake_pool, False, for_discriminator=True) * self.opt.weight_3D
                # Real Detection and Loss
                pred_real = self.netD.forward(input_B.detach())
                loss_D_real = self.criterionGAN(pred_real, True, for_discriminator=True) * self.opt.weight_3D
                # GAN loss (Fake Passability Loss)        
                pred_fake = self.netD.forward(fake_B)
                loss_G_GAN = self.criterionGAN(pred_fake, True, for_discriminator=False) * self.opt.weight_3D
            else:
                loss_D_fake = loss_D_real = loss_G_GAN = zero_tensor


            # 2D GAN loss ###########################
            pred_fake_pool_2D = self.netD_2D.forward(fake_B.detach())
            loss_2D_fake = self.criterionGAN(pred_fake_pool_2D, False, for_discriminator=True) * self.opt.weight_2D
            # Real Detection and Loss
            pred_real_2D = self.netD_2D.forward(input_B.detach())
            loss_2D_real = self.criterionGAN(pred_real_2D, True, for_discriminator=True) * self.opt.weight_2D
            # GAN loss (Fake Passability Loss)        
            pred_fake_2D = self.netD_2D.forward(fake_B)
            loss_2G_GAN = self.criterionGAN(pred_fake_2D, True, for_discriminator=False) * self.opt.weight_2D
        
            # 3d GANfeature Loss
            
            if self.opt.no_3D == False:
                if not self.opt.no_ganFeat_3D:
                    loss_3D_GAN_Feat = 0
                    feat_weights = 4.0 / (self.opt.n_layers_D + 1)
                    D_weights = 1.0 / self.opt.num_D
                    for i in range(self.opt.num_D):
                        for j in range(len(pred_fake[i])-1):
                            loss_3D_GAN_Feat += D_weights * feat_weights * \
                                self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat * self.opt.weight_3D
       
            # 2d GANfeature Loss
        
            if not self.opt.no_ganFeat_2D:
                loss_2D_GAN_Feat = 0
                feat_weights = 4.0 / (self.opt.n_layers_D + 1)
                D_weights = 1.0 / self.opt.num_D
                for i in range(self.opt.num_D):
                    for j in range(len(pred_fake_2D[i])-1):
                        loss_2D_GAN_Feat += D_weights * feat_weights * \
                            self.criterionFeat(pred_fake_2D[i][j], pred_real_2D[i][j].detach()) * self.opt.lambda_feat * self.opt.weight_2D
        
        # print(loss_3D_GAN_Feat)
        # print(loss_2D_GAN_Feat)
        """
        # VGG loss
        loss_G_VGG = zero_tensor
        if not self.opt.no_vgg:
            loss_G_VGG = 0
            loss_G_VGG += self.criterionVGG(self.reshape_tensor(fake_B), self.reshape_tensor(input_B)) * self.opt.lambda_feat * 3
            loss_G_VGG += self.criterionVGG(self.reshape_tensor(fake_B[:,:,1:2,:,:]), self.reshape_tensor(input_B[:,:,1:2,:,:])) * self.opt.lambda_feat * 7

        loss_list = [loss_2D_fake, loss_2D_real, loss_2G_GAN, loss_D_fake, loss_D_real, loss_G_GAN, loss_image, loss_2D_GAN_Feat, loss_3D_GAN_Feat, loss_G_VGG]

        # loss_list = [loss.unsqueeze(0) for loss in loss_list]           
        return loss_list

    def save(self, label):
        pass
        # if self.opt.no_3D == False:
        #     self.save_network(self.netD, 'D', label, self.gpu_ids)
        # self.save_network(self.netD_2D, 'D_2D', label, self.gpu_ids)
        
    def update_learning_rate(self, epoch):        
        if self.opt.no_SN == True:
            lr = self.opt.lr * (1 - (epoch - self.opt.niter) / self.opt.niter_decay)
        else:
            lr = self.opt.d_lr * (1 - (epoch - self.opt.niter) / self.opt.niter_decay)
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def reshape_tensor(self,input):
        assert input.dim() == 5
        this_size = input.size()
        out = input.transpose(1,2).contiguous().view(-1,this_size[1],this_size[3],this_size[4])
        assert out.dim() == 4
        assert out.size()[1] == 3
        return out

    def split_tensor(self,input):
        i_input_frame = self.opt.i_input_frame
        i_frames = self.opt.i_frames
        total_frames = input.size()[2]
        assert total_frames == (i_input_frame-1)*(i_frames+1)+1
        gen_list = []
        for index in range(total_frames):
            gen_list.append(index)

        frame_index = torch.zeros(i_input_frame).type(torch.cuda.LongTensor)
        for index in range(i_input_frame):
            frame_index[index] = index*(i_frames+1)
            gen_list.remove(index*(i_frames+1))

        gen_index = torch.zeros(len(gen_list)).type(torch.cuda.LongTensor)
        for index in range(len(gen_list)):
            gen_index[index] = gen_list[index]

        # print(frame_index)
        # print(gen_index)
        real_image = torch.index_select(input,2,frame_index)
        gen_image = torch.index_select(input,2,gen_index)
        # print(real_image.size())
        # print(gen_image.size())
        return real_image.contiguous(), gen_image.contiguous()
