import subprocess
import argparse

def run_subprocess_safely(command, exit_code=0):
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error in subprocess: {e}")
        exit(exit_code)

if __name__ == '__main__':

    # parser = argparse.ArgumentParser()   
    # parser.add_argument('--obj_name', type=str, default=None)
    # args = parser.parse_args()
    # select_name = args.obj_name

    ########## user input ##########
    select_name = 'your_object'
    instance_num = 6 # number of instances in the image. Change it to the actual number of instances in the image

    ########## hyperparameters ##########
    iters = 100 # rotation optimization iters
    absolute_mkp_threshold = 20 # threshold for absolute matching keypoint distance
    crop_size = 1200 # crop_size for each instance. 
                     # We need to crop each instance from the image for feature extraction and matching
                     # Cannot be too small, otherwise the instance may be cropped out (the code will automatically check this for you)
    train_res = 800 # training resolution for nerf. It can be different from crop_size
    rotate_delta_angle = 15 # rotation delta angle for matching pairs, total rotation times = 360 // rotate_delta_angle * instance_num ** 2

    # ------------------------------------------------------ #
    # ------------------- Preprocessing --------------------- #
    # ------------------------------------------------------ #

    ########### data path ##########
    data_folder = 'data'
    instance_dir = f'{data_folder}/{select_name}'
    image_path = f'{instance_dir}/raw/000_color.png'

    ########## run each step of the preprocessing pipeline ##########
    # run_subprocess_safely(f'python ./preprocess/0_mask_and_crop.py \
    #                       --image_path {image_path} --crop_size {crop_size} --instance_num {instance_num} --train_res {train_res}', exit_code=1)
    # run_subprocess_safely(f'python ./preprocess/1_match_pairs.py \
    #                       --instance_dir {instance_dir} --rotate_delta_angle {rotate_delta_angle} --train_res {train_res} --instance_num {instance_num}', exit_code=2)
    # run_subprocess_safely(f'python ./preprocess/2_filter_pairs.py \
    #                        --instance_dir {instance_dir} --instance_num {instance_num}', exit_code=3)
    # run_subprocess_safely(f'python ./preprocess/3_optimize_global_rotation.py \
    #                       --instance_dir {instance_dir} --instance_num {instance_num}', exit_code=4)
    # run_subprocess_safely(f'python ./preprocess/4_match_pairs_final.py \
    #                       --instance_dir {instance_dir} --instance_num {instance_num} --train_res {train_res}', exit_code=5)
    # run_subprocess_safely(f'python ./preprocess/5_sfm.py \
    #                       --instance_dir {instance_dir} --instance_num {instance_num} --train_res {train_res}', exit_code=6)
    # run_subprocess_safely(f'python ./preprocess/6_extract_sfm_point_cloud.py \
    #                       --instance_dir {instance_dir} --instance_num {instance_num} --train_res {train_res}', exit_code=7)
    run_subprocess_safely(f'python ./preprocess/7_extract_sfm_pose_and_visualize.py \
                          --instance_dir {instance_dir} --instance_num {instance_num} --train_res {train_res}', exit_code=8)
    # run_subprocess_safely(f'python ./preprocess/8_extract_monocular_cues.py \
    #                       --instance_dir {instance_dir} --instance_num {instance_num} --train_res {train_res}', exit_code=9)

    print('Preprocessing finished!')