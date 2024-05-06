import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True, help="Dataset directory")
args = parser.parse_args()

# data_all = np.load(os.path.join(args.data_dir, "data_all.pkl"), allow_pickle=True)
data_all = np.load(os.path.join(args.data_dir, "data_loftr.pkl"), allow_pickle=True)

result_dir = os.path.join(args.data_dir, "colmap_results_loftr")
img_dir = os.path.join(args.data_dir, "images_all_loftr")
assert os.path.isdir(img_dir)
for img1_name, img2_name in data_all["pair"].keys():
    out_dir = os.path.join(result_dir, f"{img1_name[:-4]}-{img2_name[:-4]}")  # remove .jpg extension
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "match_list.txt"), "w") as f:
        assert os.path.isfile(os.path.join(img_dir, img1_name))
        assert os.path.isfile(os.path.join(img_dir, img2_name))
        f.write(f"{img1_name} {img2_name}\n")

    out_img_dir = os.path.join(out_dir, "images")
    os.makedirs(out_img_dir)
    os.symlink(os.path.join(img_dir, img1_name), os.path.join(out_img_dir, img1_name))
    os.symlink(os.path.join(img_dir, img2_name), os.path.join(out_img_dir, img2_name))
    