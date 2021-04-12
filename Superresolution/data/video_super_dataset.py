### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
import random
import torch
from data.base_dataset import BaseDataset, get_img_params, get_transform, get_video_params, get_test_transform
from data.image_folder import make_grouped_dataset, make_test_grouped_dataset, check_path_valid
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

class VideoSuperDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt

        # assert self.opt.isTrain == True
        anno_file =  open(opt.anno_path, "r") 
        self.video_list = anno_file.readlines()

        self.n_of_seqs = len(self.video_list)                 # number of sequences to train       
        print("num of seqs is "+str(self.n_of_seqs))

        # self.seq_len_max = max([len(B) for B in self.B_paths])        
        if self.opt.isTrain == True:
            self.input_frames = opt.input_frames
            self.n_frames_total = self.input_frames
            assert self.n_frames_total < 31
        self.rotate = self.scale = self.shear = self.t_gamma = self.hue = 0


    def __getitem__(self, index):
        if self.opt.isTrain == True:
            B_paths = self.video_list[index % self.n_of_seqs].replace("\n","")
        
            A_paths = B_paths.replace("sequences","matlab_resize")

            total_frames = len(os.listdir(B_paths))
            if total_frames < 7:
               B_paths = self.video_list[1].replace("\n","")
               total_frames = len(os.listdir(B_paths))
               print(self.video_list[index % self.n_of_seqs].replace("\n",""))

            n_frames_total = self.n_frames_total
            t_step = 1
            

            offset_max = total_frames - 2 - n_frames_total
            # start_idx = np.random.randint(offset_max)+1

            # setting transformers
            # self.rotate = random.randint(0,3)*90

            if self.opt.use_yuv == False:
                B_img = Image.open(os.path.join(B_paths, "im1.png")).convert('RGB')        
            else:
                B_img = Image.open(os.path.join(B_paths, "im1.png")).convert('YCbCr')

            params, params2 = get_img_params(self.opt, B_img.size)          
            transform_scaleB = get_transform(self.opt, params)
            transform_scaleA = get_transform(self.opt, params2)

            self.t_gamma = random.random()
            self.hue = random.random()-0.5

            # read in images
            B = inst = 0
            A = inst = 0
            for i in range(1,8):            
                B_path = os.path.join(B_paths, "im%d.png"%(i))
                Bi = self.get_image(B_path, transform_scaleB)
                Bi = Bi.unsqueeze(1)
                B = Bi if i == 1 else torch.cat([B, Bi], dim=1)

                A_path = os.path.join(A_paths, "in%d.png"%(i))
                Ai = self.get_image(A_path, transform_scaleA)
                Ai = Ai.unsqueeze(1)
                A = Ai if i == 1 else torch.cat([A, Ai], dim=1)


            return_list = {'B': B, 'B_paths': B_path, 'A':A}

        else:
            B_paths = self.video_list[index % self.n_of_seqs].replace("\n","")
        
            A_paths = B_paths.replace("sequences","matlab_resize")

            total_frames = len(os.listdir(B_paths))
            # total_frames = 5
            n_frames_total = total_frames
            t_step = 1
            
            start_idx = self.opt.start_frame                                                   ##################
            # start_idx = 0

            # setting transformers
            if self.opt.use_yuv == False:
                B_img = Image.open(os.path.join(B_paths, "im%d.png"%(start_idx))).convert('RGB')
                A_img = Image.open(os.path.join(A_paths, "in%d.png"%(start_idx))).convert('RGB')
            else:
                B_img = Image.open(os.path.join(B_paths, "im%d.png"%(start_idx))).convert('YCbCr')
                A_img = Image.open(os.path.join(A_paths, "in%d.png"%(start_idx))).convert('YCbCr')

            transform_test_scale = get_test_transform(self.opt, B_img.size)
            transform_test_scale_in = get_test_transform(self.opt, A_img.size)

            # read in images
            B = inst = 0
            A = inst = 0
            for i in range(n_frames_total):
                B_path = os.path.join(B_paths, "im%d.png"%(start_idx+i))
                Bi = self.get_image(B_path, transform_test_scale)
                Bi = Bi.unsqueeze(1)
                B = Bi if i == 0 else torch.cat([B, Bi], dim=1)

                A_path = os.path.join(A_paths, "in%d.png"%(start_idx+i))
                Ai = self.get_image(A_path, transform_test_scale_in)
                Ai = Ai.unsqueeze(1)
                A = Ai if i == 0 else torch.cat([A, Ai], dim=1)

            return_list = {'B': B, 'B_paths': B_path, 'A':A}

        return return_list

    def get_image(self, A_path, transform_scaleA, is_label=False):
        if self.opt.use_yuv == False:
            A_img = Image.open(A_path).convert('RGB')
        else:
            A_img = Image.open(A_path).convert('YCbCr')
        ## rotate,scale,shear = random.random()-0.5, random.random()-0.5, random.random()-0.5
        # if self.opt.isTrain or self.opt.random_embed==False:
        ## A_img = transforms.functional.adjust_gamma(A_img, 0.3+0.6*random.random())
        # A_img = transforms.functional.adjust_hue(A_img, 0.14*self.hue)
        # self.rotate = 2
        # A_img = transforms.functional.affine(A_img, self.rotate,[0,0],1,0,resample=Image.BICUBIC)

        A_scaled = transform_scaleA(A_img)
        # print(A_scaled.size())
        if is_label:
            A_scaled *= 255.0
        return A_scaled

    def __len__(self):
        return len(self.video_list)

    def name(self):
        return 'VideoSuperDataset'