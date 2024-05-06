import argparse
import numpy as np
import os
import cv2
import subprocess
import shutil

from colmap_utils.database import COLMAPDatabase, pair_id_to_image_ids, blob_to_array
from colmap_utils.read_write_model import read_model, write_model
from baselines.src.colmap.utils import run_mapper, compute_metrics, dump_poses, load_from_match_list_file


def update_camera_params(db_fpath, data_all):
    db = COLMAPDatabase.connect(db_fpath)

    for img_name, camera_id in db.execute("SELECT name, camera_id FROM images"):
        assert img_name in data_all["unary"].keys()
        K = data_all["unary"][img_name]["K"]
        db.update_camera(np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]), camera_id)
    db.commit()
    db.close()


def run_colmap(img_dir, db_fpath, colmap_out_dir, data_all, colmap_path="", hide_output=False):
    if hide_output:
        pipe = subprocess.DEVNULL
    else:
        pipe = None
    subprocess.call([os.path.join(colmap_path, "colmap"), "feature_extractor",
                     "--SiftExtraction.max_num_features", "1024",
                     "--ImageReader.camera_model", "PINHOLE",
                     "--ImageReader.default_focal_length_factor", "0.85",
                    #  "--SiftExtraction.peak_threshold", "0.02",
                     "--database_path", db_fpath,
                     "--image_path", img_dir],
                     stdout=pipe, stderr=pipe)
    update_camera_params(db_fpath, data_all)
    subprocess.call([os.path.join(colmap_path, "colmap"), "exhaustive_matcher",
                     "--database_path", db_fpath,
                    #  "--SiftMatching.max_error", "3",
                    #  "--SiftMatching.min_inlier_ratio", "0.3",
                    #  "--SiftMatching.min_num_inliers", "15",
                    #  "--SiftMatching.guided_matching", "1"
                     ],
                     stdout=pipe, stderr=pipe)
    run_mapper(db_fpath, img_dir, colmap_out_dir, hide_output=hide_output)


def run_colmap_sift(exp_dir, img_dir, data_all, colmap_dir="", hide_output=True):
    out_dir = os.path.join(exp_dir, "sift")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    colmap_out_dir = os.path.join(out_dir, "sparse")
    os.makedirs(colmap_out_dir)
    db_fpath = os.path.join(out_dir, "database.db")

    match_list_fpath = os.path.join(exp_dir, "match_list.txt")
    assert os.path.isfile(match_list_fpath)

    img_names_all, matches_all = load_from_match_list_file(match_list_fpath)

    subset_img_dir = os.path.join(out_dir, "imgs")
    os.makedirs(subset_img_dir)
    for img_name in img_names_all:
        os.symlink(os.path.join(img_dir, img_name), os.path.join(subset_img_dir, img_name))

    run_colmap(subset_img_dir, db_fpath, colmap_out_dir, data_all, colmap_path=colmap_dir, hide_output=hide_output)

    return img_names_all, matches_all, colmap_out_dir
    