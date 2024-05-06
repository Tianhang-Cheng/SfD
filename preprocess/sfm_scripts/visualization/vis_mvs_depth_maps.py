import os
import numpy as np
import cv2
import argparse

from colmap_utils.read_write_dense import read_array


####################################### Parse Args ################################################
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", type=str, required=True, help="Dataset root directory")
parser.add_argument("--split", type=str, required=True, choices=["train", "validation", "test"],
    help="Dataset split, one of [train|validation|test]")
parser.add_argument("--log_id", type=str, required=True, help="Log id")
parser.add_argument("--snippet_id", type=str, required=True, help="Snippet id for the video snippet")
args = parser.parse_args()

base_dir = os.path.join(
    args.dataset_root, "snippet", args.split, args.log_id, args.snippet_id,
    "colmap_output", "dense", "stereo")
depth_dir = os.path.join(base_dir, "depth_maps")
assert(os.path.isdir(depth_dir))
output_dir = os.path.join(base_dir, "depth_maps_vis")
os.makedirs(output_dir, exist_ok=True)

for fname in sorted(os.listdir(base_dir)):
    if not "photometric" in fname:
        continue
    depth_map = read_array(os.path.join(base_dir, fname))

    min_depth, max_depth = np.percentile(
        depth_map, [5, 95])
    depth_map[depth_map < min_depth] = min_depth
    depth_map[depth_map > max_depth] = max_depth
    depth_map_normalized = (depth_map - np.min(depth_map))/(np.max(depth_map) - np.min(depth_map)) * 255.0
    cv2.imwrite(os.path.join(output_dir, fname[:6] + "_depth.png"), depth_map_normalized)