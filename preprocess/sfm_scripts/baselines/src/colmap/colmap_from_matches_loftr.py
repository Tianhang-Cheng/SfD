import argparse
import numpy as np
import os
import cv2
import shutil

from colmap_utils.database import COLMAPDatabase, pair_id_to_image_ids, blob_to_array
from baselines.src.colmap.utils import run_geometric_verification, run_mapper, load_from_match_list_file, estimate_relative_pose


def create_db_from_matches(data_all, match_list_fpath, db_fpath, img_dir, inliers=False):
    # create a database containing all images, key points and loftr matches
    if os.path.exists(db_fpath):
        # print(f"{db_fpath} already exists, removing...")
        os.remove(db_fpath)

    db = COLMAPDatabase.connect(db_fpath)
    db.create_tables()

    img_names_all, matches_all = load_from_match_list_file(match_list_fpath)
    img_name_to_id_mapping = {}
    matches_kp_inds = {}
    img_kps = {img_name: [] for img_name in img_names_all}
    for img1_name, img2_name in matches_all:
        assert (img1_name, img2_name) in data_all['pair']
        kps_matches = data_all['pair'][(img1_name, img2_name)]["kpts_loftr"]
        matches_kp_inds[(img1_name, img2_name)] = []
        for kps in kps_matches[:, :4]:
            kp1_new = kps[:2]
            kp2_new = kps[2:4]
            kp1_ind = None
            kp2_ind = None
            kp1_new_round = np.round(kp1_new, decimals=1)
            for idx, kp1 in enumerate(img_kps[img1_name]):
                if np.all(kp1_new_round == np.round(kp1, decimals=1)):
                    kp1_ind = idx
                    break
            if kp1_ind is None:
                img_kps[img1_name].append(kp1_new_round)
                kp1_ind = len(img_kps[img1_name]) - 1
            kp2_new_round = np.round(kp2_new, decimals=1)
            for idx, kp2 in enumerate(img_kps[img2_name]):
                if np.all(kp2_new_round == np.round(kp2, decimals=1)):
                    kp2_ind = idx
                    break
            if kp2_ind is None:
                img_kps[img2_name].append(kp2_new_round)
                kp2_ind = len(img_kps[img2_name]) - 1
            matches_kp_inds[(img1_name, img2_name)].append([kp1_ind, kp2_ind])
        if len(matches_kp_inds[(img1_name, img2_name)]) != 0:
            matches_kp_inds[(img1_name, img2_name)] = np.stack(matches_kp_inds[(img1_name, img2_name)])


    for img_name in img_names_all:
        # assert os.path.isfile(os.path.join(img_dir, img_name))
        img = cv2.imread(os.path.join(img_dir, img_name))
        if img is None:
            from IPython import embed
            embed()
        width, height = img.shape[1], img.shape[0]
        assert width == 640
        assert height == 480
        assert img_name in data_all["unary"].keys()
        K = data_all["unary"][img_name]["K"]
        camera_id = db.add_camera(1, width, height, np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]))  # 1 for pinhole camera
        if len(img_kps[img_name]) > 0:
            img_id = db.add_image(img_name, camera_id)
            kp = img_kps[img_name]
            kp_full = np.concatenate([kp, np.ones((len(kp), 1)), np.zeros((len(kp), 1))], axis=1).astype(np.float32)
            db.add_keypoints(img_id, kp_full)
            img_name_to_id_mapping[img_name] = img_id

    num_matches =0
    for img1_name, img2_name in matches_all:
        assert (img1_name, img2_name) in matches_kp_inds
        num_matches += len(matches_kp_inds[(img1_name, img2_name)])
        if len(matches_kp_inds[(img1_name, img2_name)]) > 0:
            sg2sp1 = matches_kp_inds[(img1_name, img2_name)][:, 0]
            sg2sp2 = matches_kp_inds[(img1_name, img2_name)][:, 1]
            kps1 = np.array(img_kps[img1_name])
            kps2 = np.array(img_kps[img2_name])
            if not inliers:
                matches = np.hstack([sg2sp1[:, None], sg2sp2[:, None]])
            else:
                ret, E = estimate_relative_pose(kps1[sg2sp1], kps2[sg2sp2], data_all["unary"][img1_name]["K"], data_all["unary"][img2_name]["K"])
                if ret is None or np.sum(ret[2]) < 15:
                    matches = np.hstack([sg2sp1[:, None], sg2sp2[:, None]])
                else:
                    # print(img1_name, img2_name, "num inliers", np.sum(ret[2]))
                    matches = np.hstack([sg2sp1[ret[2]][:, None], sg2sp2[ret[2]][:, None]])
            db.add_matches(img_name_to_id_mapping[img1_name], img_name_to_id_mapping[img2_name], matches)

    db.commit()
    db.close()

    if num_matches == 0:
        print(img_names_all)
    return img_names_all, matches_all, num_matches


def run_colmap_matches_loftr(exp_dir, img_dir, data_all, colmap_dir="", hide_output=True, inliers=False):
    dir_name = "loftr" if not inliers else "loftr_inliers"
    out_dir = os.path.join(exp_dir, dir_name)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    colmap_out_dir = os.path.join(out_dir, "sparse")
    os.makedirs(colmap_out_dir, exist_ok=True)
    db_fpath = os.path.join(out_dir, "database.db")

    match_list_fpath = os.path.join(exp_dir, "match_list.txt")
    assert os.path.isfile(match_list_fpath)

    img_names_all, matches_all, num_matches = create_db_from_matches(
        data_all, match_list_fpath, db_fpath, img_dir, inliers=inliers
    )

    if num_matches > 0:
        run_geometric_verification(db_fpath, match_list_fpath, colmap_path=colmap_dir, hide_output=hide_output)
        run_mapper(db_fpath, img_dir, colmap_out_dir, colmap_path=colmap_dir, hide_output=hide_output)
    return img_names_all, matches_all, colmap_out_dir
    