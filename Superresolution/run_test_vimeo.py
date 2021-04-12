import os
# os.system("source ~/.bashrc")
os.system("python --version")

###########################################################################

string = "python -m pdb test_eval_vimeo_m1.py --anno_path /mnt/blob/datasets/sep_testlist.txt --gpu_ids 0 " \
            "--name try_test_vimeo_save1 --input_frames 7 --batchSize 1 --nThreads 4 " \
            "--padding_mode const --G_conv_type deform --load_pretrain ./checkpoints/try_super_deform_dddn/ --resize_or_crop none --which_epoch latest"

os.system(string)


print("finished test ---- ")
