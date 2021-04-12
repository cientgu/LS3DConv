### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
import random
import torch
from data.base_dataset import BaseDataset, get_img_params, get_transform, get_video_params
from data.image_folder import make_grouped_dataset, make_test_grouped_dataset, check_path_valid
from PIL import Image
import numpy as np

class VideoPredDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        if self.opt.isTrain == True:
            self.dir_B = os.path.join(opt.dataroot, opt.phase + '_B')
            self.B_paths = make_grouped_dataset(self.dir_B)
        
            self.n_of_seqs = len(self.B_paths)                 # number of sequences to train       
            self.seq_len_max = max([len(B) for B in self.B_paths])        
            self.input_frames = opt.input_frames
            self.output_frames = opt.output_frames
            self.n_frames_total = self.input_frames + self.output_frames
            assert self.n_frames_total < 31
        else:
            self.dir_B = opt.dataroot
            self.B_paths = make_test_grouped_dataset(self.dir_B)
            
            self.n_of_seqs = len(self.B_paths)                 # number of sequences to train       
            self.seq_len_max = max([len(B) for B in self.B_paths])        
            self.input_frames = opt.input_frames
            self.output_frames = opt.output_frames
            self.n_frames_total = self.input_frames + self.output_frames
            self.n_frames_total = min(self.opt.how_many, self.seq_len_max, self.n_frames_total)

    def __getitem__(self, index):
        # print("do some  debug in temporal datasets ------------")
        tG = self.opt.n_frames_G
        B_paths = self.B_paths[index % self.n_of_seqs]                

        n_frames_total = self.n_frames_total
        t_step = 1

        if self.opt.isTrain == False:
            start_idx = self.opt.start_frame                                                   ##################
        else:
            offset_max = 30 - n_frames_total
            start_idx = np.random.randint(offset_max)

        # setting transformers
        B_img = Image.open(B_paths[start_idx]).convert('RGB')        
        params = get_img_params(self.opt, B_img.size)          
        transform_scaleB = get_transform(self.opt, params)

        # read in images
        B = inst = 0
        for i in range(n_frames_total):            
            B_path = B_paths[start_idx + i * t_step]            
            Bi = self.get_image(B_path, transform_scaleB)

            Bi = Bi.unsqueeze(1)

            B = Bi if i == 0 else torch.cat([B, Bi], dim=1)            

        return_list = {'B': B, 'B_paths': B_path}
        # A_path is something like ['datasets/Cityscapes/train_A/seq0014/aachen_000014_000010_leftImg8bit.png']
        # B_paths is something like ['datasets/Cityscapes/train_B/seq0014/aachen_000014_000010_leftImg8bit.png']
        # A size is [1,6,256,512]
        # B size is [1,18,256,512]
        # inst size is [1]


        # so A is n_frames_total * 1 channel label
        # so B is n_frames_total * 3 channel image


        return return_list

    def get_image(self, A_path, transform_scaleA, is_label=False):
        A_img = Image.open(A_path)        
        A_scaled = transform_scaleA(A_img)
        if is_label:
            A_scaled *= 255.0
        return A_scaled

    def __len__(self):
        return len(self.B_paths)

    def name(self):
        return 'VideoPredDataset'