import argparse
import numpy as np
import os
import cv2
import torch
import shutil

from tqdm import tqdm

from colmap_utils.database import COLMAPDatabase, pair_id_to_image_ids, blob_to_array
from baselines.src.colmap.utils import run_geometric_verification, run_mapper, ratio_matcher, load_from_match_list_file, \
    mutual_nn_matcher, estimate_relative_pose


def create_db_from_descriptors(data_all, match_list_fpath, db_fpath, img_dir, matcher, inliers=False):
    # create a database from descriptors
    if os.path.exists(db_fpath):
        # print(f"{db_fpath} already exists, removing...")
        os.remove(db_fpath)

    db = COLMAPDatabase.connect(db_fpath)
    db.create_tables()

    img_names_all, matches_all = load_from_match_list_file(match_list_fpath)
    img_name_to_id_mapping = {}
    for img_name in img_names_all:
        img = cv2.imread(os.path.join(img_dir, img_name))
        width, height = img.shape[1], img.shape[0]
        assert img_name in data_all["unary"].keys()
        K = data_all["unary"][img_name]["K"]
        camera_id = db.add_camera(1, width, height, np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]))  # 1 for pinhole camera
        img_id = db.add_image(img_name, camera_id)
        kp = data_all["unary"][img_name]["sp_coord"]
        kp_full = np.concatenate([kp, np.ones((len(kp), 1)), np.zeros((len(kp), 1))], axis=1).astype(np.float32)
        db.add_keypoints(img_id, kp_full)
        img_name_to_id_mapping[img_name] = img_id
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for img1_name, img2_name in matches_all:
        D1 = data_all["unary"][img1_name]["sp_desc"]
        D2 = data_all["unary"][img2_name]["sp_desc"]

        if len(D1) == 0 or len(D2) == 0:
            continue

        descriptors1 = torch.from_numpy(D1).to(device)
        descriptors2 = torch.from_numpy(D2).to(device)
        assert matcher in ["ratio", "mutual_nn"]
        if matcher == "mutual_nn":
            matches = mutual_nn_matcher(descriptors1, descriptors2).astype(np.uint32)
        else:
            matches = ratio_matcher(descriptors1, descriptors2, ratio=0.8).astype(np.uint32)
        
        if inliers:
            kps1 = data_all["unary"][img1_name]["sp_coord"]
            kps2 = data_all["unary"][img2_name]["sp_coord"]
            ret, E = estimate_relative_pose(kps1[matches[:, 0]], kps2[matches[:, 1]], data_all["unary"][img1_name]["K"], data_all["unary"][img2_name]["K"])
            if ret is not None and np.sum(ret[2]) >= 15:
                # print(f"Originally {len(matches)} matches")
                matches = matches[ret[2]]
                # print(f"Now {len(matches)} matches")
            
        # print("matches:", len(matches))
        db.add_matches(img_name_to_id_mapping[img1_name], img_name_to_id_mapping[img2_name], matches)

    db.commit()
    db.close()

    return img_names_all, matches_all


def run_colmap_desc(exp_type, exp_dir, img_dir, data_all, colmap_dir="", hide_output=True, inliers=False):
    assert(exp_type.startswith("superpoint_"))
    matcher = exp_type[11:]
    assert matcher in ["ratio", "mutual_nn"]

    dir_name = exp_type if not inliers else f"{exp_type}_inliers"

    out_dir = os.path.join(exp_dir, dir_name)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    colmap_out_dir = os.path.join(out_dir, "sparse")
    os.makedirs(colmap_out_dir, exist_ok=True)
    db_fpath = os.path.join(out_dir, "database.db")

    match_list_fpath = os.path.join(exp_dir, "match_list.txt")
    assert os.path.isfile(match_list_fpath)

    img_names_all, matches_all = create_db_from_descriptors(
        data_all, match_list_fpath, db_fpath, img_dir, matcher, inliers=inliers
    )

    run_geometric_verification(db_fpath, match_list_fpath, colmap_path=colmap_dir, hide_output=hide_output)
    run_mapper(db_fpath, img_dir, colmap_out_dir, colmap_path=colmap_dir, hide_output=hide_output)
    return img_names_all, matches_all, colmap_out_dir
    