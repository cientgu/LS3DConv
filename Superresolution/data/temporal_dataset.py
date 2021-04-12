### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
import random
import torch
from data.base_dataset import BaseDataset, get_img_params, get_transform, get_video_params
from data.image_folder import make_grouped_dataset, check_path_valid
from PIL import Image
import numpy as np

class TemporalDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + '_A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + '_B')
        self.A_is_label = self.opt.label_nc != 0

        # self.A_paths = sorted(make_grouped_dataset(self.dir_A))
        # self.B_paths = sorted(make_grouped_dataset(self.dir_B))
        self.A_paths = make_grouped_dataset(self.dir_A)
        self.B_paths = make_grouped_dataset(self.dir_B)
        check_path_valid(self.A_paths, self.B_paths)

        self.n_of_seqs = len(self.A_paths)                 # number of sequences to train       
        self.seq_len_max = max([len(A) for A in self.A_paths])        
        self.n_frames_total = self.opt.n_frames_total      # current number of frames to train in a single iteration

    def __getitem__(self, index):
        # print("do some  debug in temporal datasets ------------")
        tG = self.opt.n_frames_G
        A_paths = self.A_paths[index % self.n_of_seqs]
        B_paths = self.B_paths[index % self.n_of_seqs]                
        if self.opt.use_instance:
            I_paths = self.I_paths[index % self.n_of_seqs]                        
        
        # notice that index is index of which of 2974 seq
        # print(self.opt.use_instance)   # false
        # print(tG)                      # 3
        # print(A_paths)                 # 30 different image path from a seq  
        # print(B_paths)                 # 30 mask path from a seq


        # setting parameters
        # n_frames_total, start_idx, t_step = get_video_params(self.opt, self.n_frames_total, len(A_paths), index)   
        n_frames_total = self.n_frames_total
        t_step = 1
        offset_max = 30 - n_frames_total
        start_idx = np.random.randint(offset_max)



        # at lask, in index seq, load n_frames_total frames from start_idx and step is t_step
        # for now, each of this is 

        # print(n_frames_total)   # 6
        # print(start_idx)     # 2 or 16 or any other number
        # print(t_step)      # 1

        # setting transformers
        B_img = Image.open(B_paths[start_idx]).convert('RGB')        
        params = get_img_params(self.opt, B_img.size)          
        transform_scaleB = get_transform(self.opt, params)
        transform_scaleA = get_transform(self.opt, params, method=Image.NEAREST, normalize=False) if self.A_is_label else transform_scaleB

        # read in images
        A = B = inst = 0
        for i in range(n_frames_total):            
            A_path = A_paths[start_idx + i * t_step]
            B_path = B_paths[start_idx + i * t_step]            
            Ai = self.get_image(A_path, transform_scaleA, is_label=self.A_is_label)            
            Bi = self.get_image(B_path, transform_scaleB)

            Ai = Ai.unsqueeze(1)
            Bi = Bi.unsqueeze(1)

            A = Ai if i == 0 else torch.cat([A, Ai], dim=1)            
            B = Bi if i == 0 else torch.cat([B, Bi], dim=1)            

        return_list = {'A': A, 'B': B, 'A_path': A_path, 'B_paths': B_path}
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
        return len(self.A_paths)

    def name(self):
        return 'TemporalDataset'