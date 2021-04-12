import os
# os.system("source ~/.bashrc")
os.system("python --version")

######### this must be changed by the experiment name !!!!! ###############
string = "python -m pdb test_interpolation_foreval_g.py --anno_path /mnt/blob/datasets/gaitA/gaitA_testlist.txt --resize_or_crop none --gpu_ids 0 --name inter_deform_gaitA_short --i_input_frame 2 --i_frames 3 --batchSize 1 --padding_mode const --G_conv_type deform --manual_seed 7 --start_frame 1"
os.system(string)


print("finished training ---- ")









# python -m pdb train.py --gpu_ids 0,1 --name try_mask2vid1 --n_frames_total 8 --batchSize 2 --print_freq 1 --nThreads 2 --niter 20 --niter_decay 30 --save_epoch_freq 7 --weight_2D 2 --weight_3D 1 --weight_L2 200 --gan_mode hinge --num_D 2


# python train_prediction.py --gpu_ids 0,1 --name try3D_pred --input_frames 5 --output_frames 2 --batchSize 2 --print_freq 1 --nThreads 2 --niter 20 --niter_decay 30 --save_epoch_freq 7 --weight_2D 2 --weight_3D 1 --weight_L2 200 --gan_mode hinge --num_D 2 --padding_mode const
