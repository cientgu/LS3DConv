import torch
from torch import nn
# import torch.nn as nn

from . import vid_3d_G, vid_3d_D

def create_model(opt):    
    print(opt.model)            
    if opt.model == 'mask2vid':
        from .vid_3d_G import vid_3d_G
        modelG = vid_3d_G()
        if opt.isTrain:
            from .vid_3d_D import vid_3d_D
            modelD = vid_3d_D()
    elif opt.model == 'vidinter':
        from .vid_inter_G import vid_inter_G
        modelG = vid_inter_G()
        if opt.isTrain:
            from .vid_inter_D import vid_inter_D
            modelD = vid_inter_D()
    elif opt.model == 'vidpredict':
        from .vid_pred_G import vid_pred_G
        modelG = vid_pred_G()
        if opt.isTrain:
            from .vid_pred_D import vid_pred_D
            modelD = vid_pred_D()
    else:
        raise ValueError("Model [%s] not recognized." % opt.modelG)
    
    ###########
    # then initialize and load gpu
    modelG.initialize(opt)
    if opt.isTrain:
        modelD.initialize(opt)
        modelG = nn.DataParallel(modelG, device_ids=opt.gpu_ids)
        modelD = nn.DataParallel(modelD, device_ids=opt.gpu_ids)
        return modelG, modelD
    else:
        return modelG
