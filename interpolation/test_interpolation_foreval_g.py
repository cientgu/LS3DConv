### copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
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

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.dataset_mode = "video_inter"
opt.model = "vidinter"

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

modelG = create_model(opt)
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


print('Doing %d sequences' % len(dataset))
for i, data in enumerate(dataset):
    ####### original opt.i_input_frames = 2 and opt.i_frames = 3   #######
    print(data['B'].size())
    times = (data['B'].size()[2]-1)//(opt.i_frames+1)
    for proc_index in range(times):    
        # this_image_seq = dta['B'][:,:,proc_index*(opt.i_frames+1):(proc_index+1)*(opt.i_frames+1)+1,:,:]
        frame_index = torch.zeros(2).type(torch.LongTensor)
        for index in range(2):
            frame_index[index] = (proc_index + index)*(opt.i_frames+1)
        print(frame_index)
        input_image = torch.index_select(data['B'],2,frame_index)
        input_image = Variable(input_image)

        input_image = input_image.cuda(opt.gpu_ids[0])
        output_image = modelG.inference(input_image, save_mem=True)
            
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





# python -m pdb test_interpolation_foreval.py --gpu_ids 0 --name try_inter3_7 --i_input_frame 2 --i_frames 3 --batchSize 1 --padding_mode const --dataroot ./datasets/Cityscapes/video/stuttgart_00/ --start_frame 0 --how_many 1500
# python -m pdb test_interpolation_foreval.py --gpu_ids 0 --name try_inter_3 --i_input_frame 150 --i_frames 3 --batchSize 1 --padding_mode const --dataroot ./datasets/Cityscapes/video/stuttgart_00/ --start_frame 0 --how_many 1500  ## after change network to 83


# python -m pdb test_interpolation_foreval.py --gpu_ids 0 --name try_inter3_13 --i_input_frame 150 --i_frames 3 --batchSize 1 --padding_mode const --dataroot ./datasets/Cityscapes/video/stuttgart_00/ --start_frame 0 --how_many 1500 --G_conv_type deform
