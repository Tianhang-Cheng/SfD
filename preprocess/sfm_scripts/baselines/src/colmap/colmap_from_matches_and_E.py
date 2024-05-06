import argparse
import numpy as np
import os
import cv2
import shutil

from colmap_utils.database import COLMAPDatabase, pair_id_to_image_ids, blob_to_array
from baselines.src.colmap.utils import run_geometric_verification, run_mapper, load_from_match_list_file, estimate_relative_pose, gen_triag_input, run_triag_ba


def create_db_from_matches(data_all, match_list_fpath, db_fpath, img_dir):
    # create a database containing all images, key points and superglue matches
    if os.path.exists(db_fpath):
        # print(f"{db_fpath} already exists, removing...")
        os.remove(db_fpath)

    db = COLMAPDatabase.connect(db_fpath)
    db.create_tables()

    img_names_all, matches_all = load_from_match_list_file(match_list_fpath)
    img_name_to_id_mapping = {}
    camera_info = []
    for img_name in img_names_all:
        img = cv2.imread(os.path.join(img_dir, img_name))
        width, height = img.shape[1], img.shape[0]
        assert img_name in data_all["unary"].keys()
        K = data_all["unary"][img_name]["K"]
        camera_id = db.add_camera(1, width, height, np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]))  # 1 for pinhole camera
        camera_info.append([camera_id, width, height, K])
        assert os.path.isfile(os.path.join(img_dir, img_name))
        img_id = db.add_image(img_name, camera_id)
        kp = data_all["unary"][img_name]["sp_coord"]
        kp_full = np.concatenate([kp, np.ones((len(kp), 1)), np.zeros((len(kp), 1))], axis=1).astype(np.float32)
        db.add_keypoints(img_id, kp_full)
        img_name_to_id_mapping[img_name] = img_id

    rel_poses = {}
    use_E = True
    for img1_name, img2_name in matches_all:
        assert (img1_name, img2_name) in data_all['pair']
        sg2sp1 = data_all['pair'][(img1_name, img2_name)]["sg2sp1"]
        sg2sp2 = data_all['pair'][(img1_name, img2_name)]["sg2sp2"]
        kps1 = data_all["unary"][img1_name]["sp_coord"]
        kps2 = data_all["unary"][img2_name]["sp_coord"]
        ret, E = estimate_relative_pose(kps1[sg2sp1], kps2[sg2sp2], data_all["unary"][img1_name]["K"], data_all["unary"][img2_name]["K"])
        if ret is None or np.sum(ret[2]) == 0:
            matches = np.hstack([sg2sp1[:, None], sg2sp2[:, None]])
            use_E = False
        else:
            print(img1_name, img2_name, "num inliers", np.sum(ret[2]))
            matches = np.hstack([sg2sp1[ret[2]][:, None], sg2sp2[ret[2]][:, None]])
            rel_poses[(img1_name, img2_name)] = (ret[0], ret[1])
        db.add_matches(img_name_to_id_mapping[img1_name], img_name_to_id_mapping[img2_name], matches)

    db.commit()
    db.close()

    if use_E:
        img_exts = {}
        for img_name, img_id in img_name_to_id_mapping.items():
            if img_id == 1:
                img_exts[img_id] = np.eye(4)
                continue
            for (img1_name, img2_name), (R, t) in rel_poses.items():
                if img1_name == img_name and img_name_to_id_mapping[img2_name] in img_exts:
                    ref_ext = img_exts[img_name_to_id_mapping[img2_name]]  # world to img2
                    rel_pose = np.eye(4)  # img1_to_img2
                    rel_pose[:3, :3] = R
                    rel_pose[:3, 3] = t
                    ext = np.matmul(np.linalg.inv(rel_pose), ref_ext)  # world to img1
                    img_exts[img_id] = ext
                    break
                elif img2_name == img_name and img_name_to_id_mapping[img1_name] in img_exts:
                    ref_ext = img_exts[img_name_to_id_mapping[img1_name]]  # world to img1
                    rel_pose = np.eye(4)  # img1_to_img2
                    rel_pose[:3, :3] = R
                    rel_pose[:3, 3] = t
                    ext = np.matmul(rel_pose, ref_ext)  # world to img2
                    img_exts[img_id] = ext
                    break
            assert img_id in img_exts
        img_exts_lst = []
        for img_name, img_id in img_name_to_id_mapping.items():
            # ext = np.eye(4)
            # ext[:3, :3] = data_all['unary'][img_name]['R']
            # ext[:3, 3] = data_all['unary'][img_name]['t']
            # img_exts_lst.append([img_id, img_name, ext])
            img_exts_lst.append([img_id, img_name, img_exts[img_id]])
    else:
        img_exts_lst = None

    return img_names_all, matches_all, use_E, camera_info, img_exts_lst


def run_colmap_matches(exp_dir, img_dir, data_all, colmap_dir="", hide_output=True):
    out_dir = os.path.join(exp_dir, "superglue")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    colmap_out_dir = os.path.join(out_dir, "sparse")
    os.makedirs(colmap_out_dir, exist_ok=True)
    db_fpath = os.path.join(out_dir, "database.db")

    match_list_fpath = os.path.join(exp_dir, "match_list.txt")
    assert os.path.isfile(match_list_fpath)

    img_names_all, matches_all, use_E, camera_info, img_exts = create_db_from_matches(
        data_all, match_list_fpath, db_fpath, img_dir
    )

    run_geometric_verification(db_fpath, match_list_fpath, colmap_path=colmap_dir, hide_output=hide_output)

    if use_E:
        input_dir = os.path.join(out_dir, "input")
        gen_triag_input(input_dir, camera_info, img_exts)
        run_triag_ba(db_fpath, img_dir, input_dir, os.path.join(out_dir, "triangulated"), os.path.join(out_dir, "sparse", "0"), colmap_path=colmap_dir, hide_output=hide_output)
    else:
        run_mapper(db_fpath, img_dir, colmap_out_dir, colmap_path=colmap_dir, hide_output=hide_output)

    return img_names_all, matches_all, colmap_out_dir
    