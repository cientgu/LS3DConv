import os
# os.system("source ~/.bashrc")
os.system("python --version")

###########################################################################

string = "python -m pdb test_eval_vimeo_gen.py --anno_path /mnt/blob/datasets/vimeo_denoising_test/sep_testlist_gen.txt --gpu_ids 0 " \
            "--name try_denoise_vimeo_gen_g25_real --input_frames 7 --batchSize 1 --nThreads 4 " \
            "--padding_mode const --G_conv_type deform --load_pretrain ./checkpoints/try_denoise_real25/ --resize_or_crop none --save_image"

os.system(string)


print("finished test ---- ")
