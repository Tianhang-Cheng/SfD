"""
Description: This file contains the information of the dataset. 

Dict value description:
    instance_numbers    : the number of instances in the dataset.
    is_synthetic        : whether the dataset is synthetic or not.
    training_resolution : the resolution of the training images. Image height = Image width = training_resolution.

"""

raw_data_path = 'E:/dataset/DuplicateSingle'
processed_data_path = 'E:/dataset/DuplicateSingleImage'
blender_data_path = 'E:/dataset/Duplicate_BlenderProc'

omnidata_path='E:/code/Dup/preprocess/omnidata/omnidata_tools/torch'
pretrained_models='E:/code/Dup/preprocess/omnidata/omnidata_tools/torch/pretrained_models/' # omni data pretrained model path

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