import os
# os.system("source ~/.bashrc")
os.system("python --version")

###########################################################################

string = "python -m pdb test_eval_super.py --anno_path /mnt/blob/code/3Dvideo_g/kinetics_inter_anno_test_512.txt --gpu_ids 0,1,2,3 " \
            "--name try_super_deform --input_frames 5 --batchSize 8 --nThreads 4 " \
            "--padding_mode const --G_conv_type deform --load_pretrain ./checkpoints/try_super_deform/ --save_image"

os.system(string)


print("finished test ---- ")
