### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
import os
import numpy as np
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
import torch.nn.functional as F
import skimage
import skimage.measure

import torchvision.transforms as transforms

opt = TestOptions().parse(save=False)
opt.no_D = True
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.dataset_mode = "video_super"
opt.model = "vidsuper"

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

modelG = create_model(opt)
modelG = modelG.eval()

visualizer = Visualizer(opt)

save_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))


def reshape_concate(tensors):
    assert tensors.dim() == 4
    assert tensors.size()[0] == 1 or tensors.size()[0] == 3

    for image_index in range(tensors.size()[1]):
        if image_index == 0:
            image_tensor = tensors[:,image_index,:,:]
        else:
            this_image_tensor = tensors[:,image_index,:,:]
            image_tensor = torch.cat((image_tensor,this_image_tensor),dim=1)

    assert image_tensor.dim() == 3
    assert image_tensor.size()[1] == tensors.size()[1] * tensors.size()[2]

    return image_tensor

def PSNR(hr_images, sr_images):
    psnrs = []
    for hr_image, sr_image in zip(hr_images, sr_images):
        psnr = skimage.measure.compare_psnr(hr_image, sr_image, data_range=2)
        psnrs.append(psnr)
    return psnrs


def SSIM(hr_images, sr_images):
    ssims = []
    if np.size(hr_images, -1) == 1:
        multichannel = False
    else:
        multichannel = True
    for hr_image, sr_image in zip(hr_images, sr_images):
        if multichannel == False:
            hr_image = np.squeeze(hr_image)
            sr_image = np.squeeze(sr_image)
        ssim = skimage.measure.compare_ssim(hr_image, sr_image, data_range=2, multichannel=multichannel)
        ssims.append(ssim)
    return ssims

print('Doing %d sequences' % len(dataset))
results_path = os.path.join(opt.checkpoints_dir, opt.name, "results.txt")

totensor = transforms.ToTensor()
toimage = transforms.ToPILImage("YCbCr")
torgbimage = transforms.ToPILImage()

video_psnr = 0
video_ssim = 0
video_count = 0
for idx, data in enumerate(dataset):        # data from temporal_dataset
    this_out_dir = os.path.join(save_dir,data['B_paths'][0].split('/')[5],data['B_paths'][0].split('/')[6])
    util.mkdir(this_out_dir)

    batchSize, channel, n_frames_total, height, width = data['B'].size()  # n_frames_total = n_frames_total + tG - 1        
    # 5D tensor: batchSize, # of channels, # of frames, height, width
    # if we need to split to multi batches

    # b,c,t,h,w = data['B'].size()
    # input_image = F.avg_pool2d(data['B'].reshape(b*c,t,h,w),kernel_size=[4,4], stride=[4,4], padding=[0,0], count_include_pad=False).reshape(b,c,t,h//4,w//4)
    input_image = data['A']

    groundtruth_image = data['B']
    # print("size of input_image is ")
    # print(input_image.size())
    # print("size of ground_truth_image is ")
    # print(groundtruth_image.size())

    input_image = Variable(input_image).cuda()
    # groundtruth_image = Variable(data['B']).cuda()
    # print(data['B_paths'])

    with torch.no_grad():
        output_image = modelG(input_image)            # real_Bp is from input_B


    if opt.use_yuv == True:
        input_uv = F.avg_pool2d(data['B'][:,1:3,:,:,:].reshape(b*2,t,h,w),kernel_size=[4,4], stride=[4,4], padding=[0,0], count_include_pad=False).reshape(b,2,t,h//4,w//4)
        out_uv_tensor = F.interpolate(input_uv.reshape(b*2,t,h//4,w//4),scale_factor=4,mode='bilinear').reshape(b,2,t,h,w)
        the_output_image = torch.cat((output_image.cpu(), out_uv_tensor), dim=1)
        assert the_output_image.size()[1] == 3

        output_image = data['B'].clone()

        for frame_index in range(the_output_image.size()[2]):
            the_output_image1 = toimage(the_output_image[0,:,frame_index,:,:]/2+0.5).convert('RGB')
            the_output_image1 = totensor(the_output_image1)
            output_image[0,:,frame_index,:,:] = the_output_image1

            the_groundtruth_image1 = toimage(groundtruth_image[0,:,frame_index,:,:]/2+0.5).convert('RGB')
            the_groundtruth_image1 = totensor(the_groundtruth_image1)
            groundtruth_image[0,:,frame_index,:,:] = the_groundtruth_image1





    output_image = output_image[0].permute(1,2,3,0).cpu().detach().numpy()
    groundtruth_image = groundtruth_image[0].permute(1,2,3,0).cpu().detach().numpy()

    if opt.save_image == True:
        for this_gen_index in range(7):
            this_gen_image = output_image[this_gen_index]
            this_gen_image = np.clip((1+this_gen_image)/2.0*255.0,0,255).astype('uint8')
            util.save_image(this_gen_image, os.path.join(this_out_dir,"out_%05d.png"%(this_gen_index)))


    output_image = np.clip(output_image, -1, 1)

    if opt.no_eval == False:
        psnr_gen = PSNR(groundtruth_image, output_image)
        ssim_gen = SSIM(groundtruth_image, output_image)
        # print("[**]PSNR_gen Avg: %f, SSIM_gen Avg: %f" % (np.mean(psnr_gen), np.mean(ssim_gen)))

        this_video_psnr = np.mean(psnr_gen)
        this_video_ssim = np.mean(ssim_gen)
        video_psnr += this_video_psnr
        video_ssim += this_video_ssim
        video_count += 1

        print(idx, this_video_psnr, this_video_ssim)
        with open(results_path, 'a') as f:
            print((idx, this_video_psnr, this_video_ssim), file=f)

print("the average is :")
print(video_psnr/video_count, video_ssim/video_count)


"""
##########################
    visual_list = [('fake_image', util.tensor2im(reshape_concate(output_image[0])))]

    visuals = OrderedDict(visual_list) 
    img_path = data['B_paths']
    print('process image... %s' % img_path)
    # visualizer.save_images(save_dir, visuals, img_path)
    
    dirname = os.path.basename(os.path.dirname(img_path[0]))
    image_dir = os.path.join(save_dir, dirname)
    util.mkdir(image_dir)
    image_numpy = visual_list[0][1]
    # assert image_number == 5
    for i in range(5):
        this_image_index = proc_index*4 + i
        this_image = image_numpy[256*i:256*(i+1),0:512,:]
        print(image_dir+"/"+str(this_image_index)+".png")
        util.save_image(this_image, image_dir+"/"+str(this_image_index)+".png")


    print(save_dir)
    print(img_path)
#########################
"""
#  this code takes the whole input video in and put out the whole video and the test results





# python -m pdb test_interpolation_foreval.py --gpu_ids 0 --name try_inter3_7 --i_input_frame 2 --i_frames 3 --batchSize 1 --padding_mode const --dataroot ./datasets/Cityscapes/video/stuttgart_00/ --start_frame 0 --how_many 1500
# python -m pdb test_interpolation_foreval.py --gpu_ids 0 --name try_inter_3 --i_input_frame 150 --i_frames 3 --batchSize 1 --padding_mode const --dataroot ./datasets/Cityscapes/video/stuttgart_00/ --start_frame 0 --how_many 1500  ## after change network to 83


# python -m pdb test_interpolation_foreval.py --gpu_ids 0 --name try_inter3_13 --i_input_frame 150 --i_frames 3 --batchSize 1 --padding_mode const --dataroot ./datasets/Cityscapes/video/stuttgart_00/ --start_frame 0 --how_many 1500 --G_conv_type deform
