import os
import cv2
from tqdm import tqdm

root_dir = "/Users/joyceyang/Documents/generalized_sfm/matching/"
splits = ["for_joyce_haggling", "for_joyce_sg2", "for_joyce_sg_dance_2", "for_joyce_mc", "for_joyce_mc_ours"]

for split in splits:
    img_dir = os.path.join(root_dir, split, "images_all")
    loftr_img_dir = os.path.join(root_dir, split, "images_all_loftr")
    os.makedirs(loftr_img_dir, exist_ok=True)
    print(f"Processing {split}")
    for img_name in tqdm(os.listdir(img_dir)):
        img_raw = cv2.imread(os.path.join(img_dir, img_name))
        img_raw = cv2.resize(img_raw, (640, 480))
        cv2.imwrite(os.path.join(loftr_img_dir, img_name), img_raw)