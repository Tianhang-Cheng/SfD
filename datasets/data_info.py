"""
Description: This file contains the information of the dataset.

Dict value description:
    instance_numbers    : the number of instances in the dataset.
    is_synthetic        : whether the dataset is synthetic or not.
    training_resolution : the resolution of the training images. Image height = Image width = training_resolution.

Paths: the four *_path values below are only used by the authors' own dataset-building and
Blender scripts, and by preprocess/8_extract_monocular_cues.py. They default to folders inside this
repository (data/, hf_data/, blender_data/ -- the same names download_assets.py and the README use),
so nothing has to be edited after cloning. Point them anywhere else with SFD_RAW_DATA_PATH /
SFD_PROCESSED_DATA_PATH / SFD_BLENDER_DATA_PATH. The Omnidata paths also resolve relative to this
repository, so the vendored copy in preprocess/omnidata is found no matter where you cloned it.
"""

import os

_REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# raw single-image inputs, one folder per object with raw/000_rgb.{png,exr} inside
raw_data_path = os.environ.get('SFD_RAW_DATA_PATH', os.path.join(_REPO_DIR, 'data'))
# the pre-processed DuplicateSingleImage dataset (what `hf download` writes)
processed_data_path = os.environ.get('SFD_PROCESSED_DATA_PATH', os.path.join(_REPO_DIR, 'hf_data'))
# the DuplicateBlenderData scenes/renders (what `python download_assets.py --blender-data` writes)
blender_data_path = os.environ.get('SFD_BLENDER_DATA_PATH', os.path.join(_REPO_DIR, 'blender_data'))

# preprocess/omnidata is a trimmed copy of https://github.com/EPFL-VILAB/omnidata that ships with
# this repository; only the omnidata_dpt_normal_v2.ckpt checkpoint has to be downloaded by hand.
omnidata_path = os.path.join(_REPO_DIR, 'preprocess', 'omnidata', 'omnidata_tools', 'torch')
pretrained_models = os.path.join(omnidata_path, 'pretrained_models')  # omni data pretrained model path

obj_info = {

    # test pipeline
    'test'      :[10, False, 800],

    # ablation study (change number of instances)
    'box2'      :[2, True, 800],
    'box4'      :[4, True, 800],
    'box6'      :[6, True, 800],
    'box8'      :[8, True, 800],
    'box10'     :[10, True, 800],
    'box15'     :[15, True, 800],
    'box20'     :[20, True, 800],
    'box25'     :[25, True, 800],
    'box30'     :[30, True, 800],
    'box40'     :[40, True, 800],
    'box50'     :[50, True, 800],
    'box60'     :[60, True, 800],

    # ablation study (change representation)
    'cash0'    :[10, True, 800], # our
    'cash1'    :[10, True, 800], # hessian
    'cash2'    :[10, True, 800], # colmap
    'cash3'    :[10, True, 800], # pose optimization
    'cash4'    :[10, True, 800], # triplane
    'cash5'    :[10, True, 800], # hash-based MLP + numerical gradient
    'cash6'    :[10, True, 800], # hash-based MLP + analytical gradient

    # synthetic single view
    'box'       :[10, True, 800],
    'cash'      :[10, True, 800],
    'cleaner'   :[9, True, 800],
    'clock'     :[9, True, 800],
    'coffee'    :[7, True, 800],
    'fire'      :[10, True, 800],
    'gitar'     :[9, True, 800],
    'sign'      :[10, True, 800],
    'tin'       :[9, True, 800],
    'paint'     :[70, True, 800],

    # test pipeline
    'monkey'    : [8, False, 800],

    # real-world single view
    'cheese'    :[5, False, 800],
    'yogurt'    :[10, False, 800],
    'airplane'  :[6, False, 800],
    'cola'      :[7, False, 800],
    'cake'      :[7, False, 800],
    'potato'    :[9, False, 800],
    'crane'     :[28, False, 1588]
}

# bounding box scale
scales = {
    'box_single':[1.5,2 ,4],
    'cash': [2.8,2.4,2.6],
    'cleaner': [1.9,1.2 ,4],
    'clock': [1.9,1.2 ,2.5],
    'coffee': [3,3,3],
    'fire':[2.1,2.1,4],
    'gitar': [2.2,0.8,6],
    'sign':[2,1.2 ,3.5],
    'tin':[3,3,3],
    'airplane': [1,1,1]
}