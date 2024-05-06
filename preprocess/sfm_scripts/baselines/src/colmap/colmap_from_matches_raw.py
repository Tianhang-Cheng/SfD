import argparse
import numpy as np
import os
import cv2
import shutil

from colmap_utils.database import COLMAPDatabase, pair_id_to_image_ids, blob_to_array
from baselines.src.colmap.utils import run_geometric_verification, run_mapper, load_from_match_list_file, estimate_relative_pose


def create_db_from_matches(data_all, match_list_fpath, db_fpath, img_dir, raw_match_fpath):
    # create a database containing all images, key points and superglue matches
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
        assert os.path.isfile(os.path.join(img_dir, img_name))
        img_id = db.add_image(img_name, camera_id)
        kp = data_all["unary"][img_name]["sp_coord"]
        kp_full = np.concatenate([kp, np.ones((len(kp), 1)), np.zeros((len(kp), 1))], axis=1).astype(np.float32)
        db.add_keypoints(img_id, kp_full)
        img_name_to_id_mapping[img_name] = img_id

    with open(raw_match_fpath, "w") as f:
        for img1_name, img2_name in matches_all:
            assert (img1_name, img2_name) in data_all['pair']
            sg2sp1 = data_all['pair'][(img1_name, img2_name)]["sg2sp1"]
            sg2sp2 = data_all['pair'][(img1_name, img2_name)]["sg2sp2"]
            kps1 = data_all["unary"][img1_name]["sp_coord"]
            kps2 = data_all["unary"][img2_name]["sp_coord"]
            ret, E = estimate_relative_pose(kps1[sg2sp1], kps2[sg2sp2], data_all["unary"][img1_name]["K"], data_all["unary"][img2_name]["K"])
            # if True:
            if ret is None or np.sum(ret[2]) == 0:
                matches = np.hstack([sg2sp1[:, None], sg2sp2[:, None]])
            else:
                print(img1_name, img2_name, "num inliers", np.sum(ret[2]))
                matches = np.hstack([sg2sp1[ret[2]][:, None], sg2sp2[ret[2]][:, None]])
            f.write("{} {}\n".format(img1_name, img2_name))
            for idx1, idx2 in matches:
                f.write("{} {}\n".format(idx1, idx2))
            f.write("\n")
            # db.add_matches(img_name_to_id_mapping[img1_name], img_name_to_id_mapping[img2_name], matches)

    db.commit()
    db.close()

    return img_names_all, matches_all


def run_colmap_matches(exp_dir, img_dir, data_all, colmap_dir="", hide_output=True):
    out_dir = os.path.join(exp_dir, "superglue")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    colmap_out_dir = os.path.join(out_dir, "sparse")
    os.makedirs(colmap_out_dir, exist_ok=True)
    db_fpath = os.path.join(out_dir, "database.db")

    match_list_fpath = os.path.join(exp_dir, "match_list.txt")
    raw_match_fpath = os.path.join(out_dir, "match_list.txt")
    assert os.path.isfile(match_list_fpath)

    img_names_all, matches_all = create_db_from_matches(
        data_all, match_list_fpath, db_fpath, img_dir, raw_match_fpath
    )

    run_geometric_verification(db_fpath, raw_match_fpath, colmap_path=colmap_dir, hide_output=hide_output, type="inliers")
    # run_geometric_verification(db_fpath, raw_match_fpath, colmap_path=colmap_dir, hide_output=hide_output, type="raw")
    run_mapper(db_fpath, img_dir, colmap_out_dir, colmap_path=colmap_dir, hide_output=hide_output)
    return img_names_all, matches_all, colmap_out_dir
    