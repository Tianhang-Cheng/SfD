import os
import argparse
import itertools


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True, help="Dataset directory")
args = parser.parse_args()

with open(os.path.join(args.data_dir, "template.txt"), "r") as fd:
    lines = [f.strip().split() for f in fd.readlines()]

result_dir = os.path.join(args.data_dir, "colmap_results_loftr")
img_dir = os.path.join(args.data_dir, "images_all_loftr")
assert os.path.isdir(img_dir)
max_num_views = 5
for line in lines:
    template = line[0].split("/")
    template = f"{template[-4]}_{template[-1]}"
    out_dir = os.path.join(result_dir, f"{template[:-4]}")  # remove .jpg extension
    os.makedirs(out_dir, exist_ok=True)
    pairs_all = list(itertools.combinations(line[1:][::-1], 2))[::-1]
    img_names_all = []
    for n in range(2, max_num_views + 1):
        out_dir_per_n = os.path.join(out_dir, f"n_{n:03d}")
        os.makedirs(out_dir_per_n, exist_ok=True)
        with open(os.path.join(out_dir_per_n, "match_list.txt"), "w") as f:
            for pair_idx in range(n*(n-1)//2):
                cam_idx_2, cam_idx_1 = list(map(int, pairs_all[pair_idx]))
                img1_name = template.replace("_01_", f"_{cam_idx_1:02d}_")
                img2_name = template.replace("_01_", f"_{cam_idx_2:02d}_")
                assert os.path.isfile(os.path.join(img_dir, img1_name)), f"{os.path.join(img_dir, img1_name)} does not exist"
                assert os.path.isfile(os.path.join(img_dir, img2_name)), f"{os.path.join(img_dir, img2_name)} does not exist"
                f.write(f"{img1_name} {img2_name}\n")
                if n == max_num_views:
                    img_names_all.append(img1_name)
                    img_names_all.append(img2_name)

    img_names_all = sorted(list(set(img_names_all)))

    out_img_dir = os.path.join(out_dir, "images")
    os.makedirs(out_img_dir, exist_ok=True)
    for img_name in img_names_all:
        os.symlink(os.path.join(img_dir, img_name), os.path.join(out_img_dir, img_name))
    