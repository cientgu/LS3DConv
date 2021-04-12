### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
import random
import torch
from data.base_dataset import BaseDataset, get_img_params, get_transform, get_video_params
from data.image_folder import make_grouped_dataset, make_test_grouped_dataset, check_path_valid
from PIL import Image
import numpy as np

class VideoInterDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt

        assert self.opt.isTrain == True
        anno_file =  open(opt.anno_path, "r") 
        self.video_list = anno_file.readlines()

        # self.dir_B = os.path.join(opt.dataroot, opt.phase + '_B')
        # self.B_paths = make_grouped_dataset(self.dir_B)
    
        self.n_of_seqs = len(self.video_list)                 # number of sequences to train       
        # self.seq_len_max = max([len(B) for B in self.B_paths])        
        self.i_input_frame = opt.i_input_frame
        self.i_frames = opt.i_frames
        self.n_frames_total = (self.i_frames + 1)*(self.i_input_frame - 1) + 1
        assert self.n_frames_total < 31

    def __getitem__(self, index):
        # print("do some  debug in temporal datasets ------------")

        B_paths = self.video_list[index % self.n_of_seqs].replace("\n","")

        total_frames = len(os.listdir(B_paths))
        n_frames_total = self.n_frames_total
        t_step = 1
        
        if self.opt.isTrain == False:
            start_idx = self.opt.start_frame                                                   ##################
        else:
            offset_max = total_frames - 2 - n_frames_total
            start_idx = np.random.randint(offset_max)+1

        # setting transformers
        B_img = Image.open(os.path.join(B_paths, "image_%05d.jpg"%(start_idx))).convert('RGB')        

        width, height = B_img.size
        left=(width-self.opt.loadSize)//2
        right = left+self.opt.loadSize
        top =(height-self.opt.loadSize)//2
        bottom = top+self.opt.loadSize
        B_img = B_img.crop((left,top,right,bottom))

        params = get_img_params(self.opt, B_img.size)          
        transform_scaleB = get_transform(self.opt, params)

        # read in images
        B = inst = 0
        for i in range(n_frames_total):            
            B_path = os.path.join(B_paths, "image_%05d.jpg"%(start_idx+i)) 
            Bi = self.get_image(B_path, transform_scaleB)
            Bi = Bi.unsqueeze(1)
            B = Bi if i == 0 else torch.cat([B, Bi], dim=1)

        return_list = {'B': B, 'B_paths': B_path}

        return return_list

    def get_image(self, A_path, transform_scaleA, is_label=False):
        A_img = Image.open(A_path)        
        A_scaled = transform_scaleA(A_img)
        if is_label:
            A_scaled *= 255.0
        return A_scaled

    def __len__(self):
        return len(self.video_list)

    def name(self):
        return 'VideoInterDataset'