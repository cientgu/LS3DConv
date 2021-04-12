### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
import random
import torch
from data.base_dataset import BaseDataset, get_img_params, get_transform, get_video_params
from data.image_folder import make_grouped_dataset, make_test_grouped_dataset, check_path_valid
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

class VideoInterDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.use_yuv = opt.use_yuv
        # assert self.opt.isTrain == True
        anno_file =  open(opt.anno_path, "r") 
        self.video_list = anno_file.readlines()

        self.n_of_seqs = len(self.video_list)                 # number of sequences to train       
        # print("num of seqs is "+str(self.n_of_seqs))

        # self.seq_len_max = max([len(B) for B in self.B_paths])        
        if self.opt.isTrain == True:
            self.i_input_frame = opt.i_input_frame
            self.i_frames = opt.i_frames
            self.n_frames_total = (self.i_frames + 1)*(self.i_input_frame - 1) + 1
            # assert self.n_frames_total < 8
        self.rotate = self.scale = self.shear = self.t_gamma = self.hue = 0


    def __getitem__(self, index):
        if self.opt.isTrain == True:
            B_paths = self.video_list[index % self.n_of_seqs].replace("\n","")
            
            this_list = os.listdir(B_paths)
            total_frames = len(this_list)
            """ 
            if total_frames < 7:
               B_paths = self.video_list[1].replace("\n","")
               total_frames = len(os.listdir(B_paths))
               print(self.video_list[index % self.n_of_seqs].replace("\n",""))
            """
            n_frames_total = self.n_frames_total
            t_step = 1
            
            # print(total_frames, n_frames_total)
            offset_max = total_frames - n_frames_total
            # start_idx = np.random.randint(offset_max)+1

            this_list = sorted(this_list, key=lambda x:int(x[2:-4]))
            first_num = int(this_list[0].replace(".png","").replace("im",""))
            # print(first_num)
            start_idx = first_num+np.random.randint(offset_max)
            # print(start_idx)            

            # setting transformers
            # try:
            if self.opt.use_yuv == False:
                B_img = Image.open(os.path.join(B_paths, "im%d.png"%(first_num))).convert('RGB')        
            else:
                B_img = Image.open(os.path.join(B_paths, "im%d.png"%(first_num))).convert('YCbCr')

            params = get_img_params(self.opt, B_img.size)          
            transform_scaleB = get_transform(self.opt, params)

            self.rotate = random.random()-0.5
            self.scale = random.random()-0.5
            self.shear = random.random()-0.5
            self.t_gamma = random.random()
            self.hue = random.random()-0.5

            # read in images
            B = inst = 0
            # start_idx = random.randint(1,offset_max)
            # print(start_idx)
            # print("----") 
            for i in range(start_idx,start_idx+n_frames_total):            
                # print(i)
                B_path = os.path.join(B_paths, "im%d.png"%(i))
                # print(B_path)
                Bi = self.get_image(B_path, transform_scaleB)
                Bi = Bi.unsqueeze(1)
                # print(Bi.size())
                B = Bi if i == start_idx else torch.cat([B, Bi], dim=1)

            return_list = {'B': B, 'B_paths': B_path}

        else:
            B_paths = self.video_list[index % self.n_of_seqs].replace("\n","")
        
            this_list = os.listdir(B_paths)
            total_frames = len(this_list)
            # total_frames = 5
            n_frames_total = total_frames
            t_step = 1
            
            # this_list = sorted(this_list, key=lambda x:int(x[2:-4]))
            # start_idx = int(this_list[0].replace(".png","").replace("im",""))
            start_idx = self.opt.start_frame                                                   ##################
            # start_idx = 0


            # setting transformers
            if self.opt.use_yuv == False:
                B_img = Image.open(os.path.join(B_paths, "im%d.png"%(start_idx))).convert('RGB')
            else:
                B_img = Image.open(os.path.join(B_paths, "im%d.png"%(start_idx))).convert('YCbCr')

            params = get_img_params(self.opt, B_img.size)
            transform_test_scale = get_transform(self.opt, params)

            # read in images
            B = inst = 0
            for i in range(n_frames_total):
                B_path = os.path.join(B_paths, "im%d.png"%(start_idx+i))
                Bi = self.get_image(B_path, transform_test_scale)
                Bi = Bi.unsqueeze(1)
                B = Bi if i == 0 else torch.cat([B, Bi], dim=1)

            return_list = {'B': B, 'B_paths': B_path}


        return return_list

    def get_image(self, A_path, transform_scaleA, is_label=False):
        if self.opt.use_yuv == False:
            A_img = Image.open(A_path).convert('RGB')
        else:
            A_img = Image.open(A_path)

        ## rotate,scale,shear = random.random()-0.5, random.random()-0.5, random.random()-0.5
        # if self.opt.isTrain or self.opt.random_embed==False:
        ## A_img = transforms.functional.adjust_gamma(A_img, 0.3+0.6*random.random())
        A_img = transforms.functional.adjust_hue(A_img, 0.07*self.hue)
        # A_img = transforms.functional.affine(A_img, 5*self.rotate,[0,0],1+0.1*self.scale,3*self.shear,resample=Image.BICUBIC)

        A_scaled = transform_scaleA(A_img)
        if is_label:
            A_scaled *= 255.0
        return A_scaled

    def __len__(self):
        return len(self.video_list)

    def name(self):
        return 'VideoInterDataset'
