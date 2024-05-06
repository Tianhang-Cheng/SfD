import argparse
import os
import numpy as np
import cv2

from tqdm import tqdm

# from baselines.src.colmap.colmap_from_matches_and_E_raw import run_colmap_matches
# from baselines.src.colmap.colmap_from_matches_and_E import run_colmap_matches
# from baselines.src.colmap.colmap_from_matches_raw import run_colmap_matches
from baselines.src.colmap.colmap_from_matches import run_colmap_matches
from baselines.src.colmap.colmap_from_matches_loftr import run_colmap_matches_loftr
from baselines.src.colmap.colmap_from_descriptors import run_colmap_desc
from baselines.src.colmap.colmap_with_sift import run_colmap_sift
from colmap_utils.read_write_model import read_model, write_model
from baselines.src.colmap.utils import load_from_match_list_file, dump_poses, compute_metrics, vis_poses_3D_with_gt


def main():
    ####################################### Parse Args ################################################
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset directory")
    parser.add_argument("--exp_type", type=str, required=True,
        choices=["superglue", "sift", "superpoint_mutual_nn", "superpoint_ratio", "loftr"], help="Exp type")
    parser.add_argument("--colmap_dir", type=str, default="", help="Dir containing the colmap executable")
    parser.add_argument("--template_name", type=str, default=None, help="Template name, e.g., 171204_pose3_00_01_00000383")
    parser.add_argument("--template_name_list_fpath", type=str, default=None, help="Path to a list of template names,\
        e.g., each line containing something like 171204_pose3_00_01_00000383")
    parser.add_argument("--display_colmap_output", action="store_true", default=False, help="Whether to display colmap output")
    parser.add_argument("--inliers", action="store_true", default=False, help="Whether to use inliers from 8-point algorithm")
    args = parser.parse_args()

    cv2.setRNGSeed(626)

    if args.exp_type == "loftr":
        results_dir = os.path.join(args.data_dir, "colmap_results_loftr")
        data_all = np.load(os.path.join(args.data_dir, "data_loftr.pkl"), allow_pickle=True)
    else:
        results_dir = os.path.join(args.data_dir, "colmap_results")
        data_all = np.load(os.path.join(args.data_dir, "data_all.pkl"), allow_pickle=True)
    assert os.path.isdir(results_dir)
    
    if args.template_name is None and args.template_name_list_fpath is None:
        all_groups = list(sorted(os.listdir(results_dir)))
    elif args.template_name_list_fpath is None:
        all_groups = [args.template_name]
    else:
        assert os.path.isfile(args.template_name_list_fpath)
        with open(args.template_name_list_fpath, "r") as f:
            lines = f.readlines()
            all_groups = [line.rstrip() for line in lines]

    # all_groups = ["0a312f741fdf5d89-two_teachers_desk_1_20_shorter_002844-0a312f741fdf5d89-two_teachers_desk_1_20_shorter_002866"]
    for group in tqdm(all_groups):
        # print(f"Processing group {group}")
        group_dir = os.path.join(results_dir, group)

        img_dir = os.path.join(group_dir, "images")
        assert os.path.isdir(img_dir)

        if args.exp_type == "superglue":
            img_names_all, matches_all, colmap_out_dir = run_colmap_matches(
                group_dir, img_dir, data_all, colmap_dir=args.colmap_dir, hide_output=(not args.display_colmap_output), inliers=args.inliers)
        elif args.exp_type == "loftr":
            img_names_all, matches_all, colmap_out_dir = run_colmap_matches_loftr(
                group_dir, img_dir, data_all, colmap_dir=args.colmap_dir, hide_output=(not args.display_colmap_output), inliers=args.inliers)
        elif args.exp_type.startswith("superpoint"):
            img_names_all, matches_all, colmap_out_dir = run_colmap_desc(
                args.exp_type, group_dir, img_dir, data_all, colmap_dir=args.colmap_dir, hide_output=(not args.display_colmap_output), inliers=args.inliers)
        elif args.exp_type == "sift":
            img_names_all, matches_all, colmap_out_dir = run_colmap_sift(
                group_dir, img_dir, data_all, colmap_dir=args.colmap_dir, hide_output=(not args.display_colmap_output))
        else:
            raise ValueError(f"Unknown exp type {args.exp_type}")

        if not os.path.isfile(os.path.join(colmap_out_dir, "0", "images.bin")):
            # print("ERROR: COLMAP FAILED!")
            with open(os.path.join(colmap_out_dir, "!COLMAP_FAILED"), "w") as f:
                pass
        else:
            cameras, images, points3D = read_model(path=os.path.join(colmap_out_dir, "0"), ext=".bin")
            write_model(cameras, images, points3D, path=os.path.join(colmap_out_dir, "0"), ext=".txt")
            poses_est = dump_poses(
                os.path.join(colmap_out_dir, "0", "images.txt"),
                img_names_all,
                os.path.join(colmap_out_dir, "poses.npz")
            )
            metrics_results = compute_metrics(poses_est, matches_all, data_all)
            np.savez(os.path.join(colmap_out_dir, "poses_metrics.npz"), metrics_results)
            # vis_poses_3D_with_gt(poses_est, img_names_all, data_all, pcd_fpath=os.path.join(colmap_out_dir, "0", "points3D.txt"))


if __name__ == "__main__":
    main()