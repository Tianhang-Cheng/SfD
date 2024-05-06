import argparse
import numpy as np
import os
import cv2
import shutil

from tqdm import tqdm

from baselines.src.colmap.utils import load_from_match_list_file, compute_metrics


def main():
    ####################################### Parse Args ################################################
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset directory")
    parser.add_argument("--exp_type", type=str, required=True, choices=[
        "superglue", "superglue_inliers", "sift", "superpoint_mutual_nn", "superpoint_mutual_nn_inliers", "superpoint_ratio"
    ], help="Exp type")
    args = parser.parse_args()

    results_dir = os.path.join(args.data_dir, "colmap_results_inliers")
    assert os.path.isdir(results_dir)

    print(f"Processing baseline exp type {args.exp_type}")

    # data_all = np.load(os.path.join(args.data_dir, "data_part_0.pkl"), allow_pickle=True)
    # tmp_data = np.load(os.path.join(args.data_dir, "data_part_1.pkl"), allow_pickle=True)
    # data_all["unary"].update(tmp_data["unary"])
    # data_all["pair"].update(tmp_data["pair"])
    # tmp_data = np.load(os.path.join(args.data_dir, "data_part_2.pkl"), allow_pickle=True)
    # data_all["unary"].update(tmp_data["unary"])
    # data_all["pair"].update(tmp_data["pair"])
    
    all_groups = list(sorted(os.listdir(results_dir)))
    
    missing_dirs = []
    failed_dirs = []
    incomplete_pose_dirs = []
    for group in tqdm(all_groups):
        # print(f"Processing group {group}")
        group_dir = os.path.join(results_dir, group)

        experiments_dir_lst = []
        # for n in range(7, 11):
        for n in range(2, 7):
            exp_dir = os.path.join(group_dir, f"n_{n:03d}")
            assert os.path.isdir(exp_dir)
            experiments_dir_lst.append(exp_dir)

        for exp_dir in experiments_dir_lst:
            # print(f"Processing exp_dir {exp_dir}")
            out_dir = os.path.join(exp_dir, args.exp_type)
            colmap_out_dir = os.path.join(out_dir, "sparse")
            
            if not os.path.isdir(out_dir) or not os.path.isdir(colmap_out_dir):
                # print(f"Missing: {out_dir}")
                missing_dirs.append(out_dir)
                continue

            if os.path.isfile(os.path.join(colmap_out_dir, "!COLMAP_FAILED")):
                # print(f"COLMAP failed: {out_dir}")
                failed_dirs.append(out_dir)
                continue

            if not os.path.isfile(os.path.join(colmap_out_dir, "0", "images.bin")):
                # print(f"Missing: {out_dir}")
                missing_dirs.append(out_dir)
                continue

            assert os.path.isfile(os.path.join(colmap_out_dir, "poses.npz"))
            assert os.path.isfile(os.path.join(colmap_out_dir, "poses_metrics.npz"))

            # check whether the poses are complete
            image_names_all, matches_all = load_from_match_list_file(os.path.join(exp_dir, "match_list.txt"))
            dumped_poses = np.load(os.path.join(colmap_out_dir, "poses.npz"), allow_pickle=True)
            if not sorted(dumped_poses["arr_0"].item().keys()) == sorted(image_names_all):
                # print(f"Incomplete poses: {out_dir}")
                incomplete_pose_dirs.append(out_dir)
            # compute_metrics(dumped_poses["arr_0"].item(), matches_all, data_all)
    
    print("Total number of exps:", len(all_groups) * 5)
    print("Missing exps:", len(missing_dirs))
    print("Failed exps:", len(failed_dirs))
    print("Incomplete pose exps:", len(incomplete_pose_dirs))

    from IPython import embed
    embed()


if __name__ == "__main__":
    main()
    