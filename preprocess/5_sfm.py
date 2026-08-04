import sys
import os
import argparse

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
from pathlib import Path 
import pycolmap

from sfm_util.utils import estimation_and_geometric_verification, run_reconstruction 
from sfm_scripts.colmap_utils.database import COLMAPDatabase
from sfm_scripts.colmap_utils.read_write_model import read_model, write_model
from sfm_scripts.baselines.src.colmap.utils import dump_poses

if __name__ == '__main__':

    parser = argparse.ArgumentParser()   
    
    parser.add_argument(
        '--train_res', type=int, default=800, help='training resolution for nerf')
    parser.add_argument(
        '--instance_num', type=int, default=7, required=True, help='number of instances in the image')
    parser.add_argument(
        '--instance_dir', type=str, default=None, help='instance directory'
    )

    args = parser.parse_args()

    train_res = args.train_res
    instance_num = args.instance_num
    instance_dir = args.instance_dir

    raw_dir = os.path.join(instance_dir, 'raw')
    temp_dir = os.path.join(raw_dir, 'temp')
    resized_dir = os.path.join(temp_dir, 'resized')
    sfm_input_dir = os.path.join(temp_dir, 'sfm_inputs')
    sfm_output_dir = os.path.join(temp_dir, 'sfm_outputs')

    images_dir = Path(resized_dir)

    sfm_dir = Path(sfm_output_dir)
    sfm_dir.mkdir(exist_ok=True, parents=True)
    database_path = sfm_dir / 'database.db'

    matches_path = os.path.join(sfm_input_dir, 'final_matches.npz')
    feats_path = os.path.join(sfm_input_dir, 'final_feats.npz')
    pair_path = os.path.join(sfm_input_dir, 'image_pair.txt')

    # camera
    n = instance_num
    h = w = train_res
    f = 1111 # initial guess of focal length 

    if os.path.exists(database_path):
        os.remove(database_path)

    # create db
    print('Create database ...')
    db =  COLMAPDatabase.connect(database_path)
    db.create_tables()

    f_feats = dict(np.load(feats_path, allow_pickle=True))
    f_matches = dict(np.load(matches_path, allow_pickle=True))
    if not os.path.exists(pair_path): 
        os.makedirs(os.path.dirname(pair_path), exist_ok=True)
        f_pairs = open(pair_path, 'w')
        from itertools import combinations
        lines = list(combinations(range(0, n), 2))
        for line in lines:
            name1 = '{}_rgb.png'.format(str(line[0]).zfill(3))
            name2 = '{}_rgb.png'.format(str(line[1]).zfill(3))
            f_pairs.writelines('{} {}\n'.format(name1, name2))
        f_pairs.close()
    f_pairs = open(pair_path, 'r')  

    # add camera
    model, width, height, camera_params = 0, h, w, np.array((f, h/2-0.5, w/2-0.5)) # same intrinsic
    camera_id = db.add_camera(model, width, height, camera_params)
    
    obj_name_to_image_ids = {}
    image_names = []

    options = { }

    bad_obj = [ ]

    # stage 4 only writes features for instances that survived in at least one good pair.
    # An instance missing here can never be registered, so record it instead of crashing.
    no_feature_obj = []

    for i in range(n):

        image_id = db.add_image("{}_rgb.png".format(str(i).zfill(3)), camera_id)
        image_names.append("{}_rgb.png".format(str(i).zfill(3)))
        print('add "{}_rgb.png", image_id = {}'.format(str(i).zfill(3), image_id))
        obj_name_to_image_ids[i] = image_id
        # pdb.set_trace()
        if i in bad_obj:
            continue
        if str(i) not in f_feats:
            no_feature_obj.append(i)
            print('instance {} has no features in final_feats.npz '
                  '(stage 2/4 found no good pair for it), skip.'.format(i))
            continue
        db.add_keypoints(image_id, f_feats[str(i)].item()['keypoints_back'])

    print(image_names)

    bad_pair = []
    if f_pairs is not None:
        lines = f_pairs.readlines()

    num_matches_per_obj = {i: 0 for i in range(n)}
    num_pairs_per_obj = {i: 0 for i in range(n)}

    for line in lines:
        name1, name2 = line.split()[0:2]
        image_name1 = int(name1[0:3])
        image_name2 = int(name2[0:3])

        if (image_name1, image_name2) in bad_pair :
            continue

        if image_name1 in no_feature_obj or image_name2 in no_feature_obj:
            print('skip pair {} - {}: one side has no features.'.format(image_name1, image_name2))
            continue

        print('add match {} - {}'.format(image_name1, image_name2))

        match_key = '{}_{}'.format(str(image_name1), str(image_name2))
        if match_key not in f_matches:
            print(match_key, ' is missed.')
            continue
        matches = f_matches[match_key].item()['matches0'].copy()
        valid = matches > -1
        matches1 = np.where(valid == True)[0]
        matches2 = matches[valid]
        # matches2 = f_matches['{}_{}'.format(str(image_id1), str(image_id2))].item()['matches1'].copy()
        # matches2 = matches2[matches2 > -1]
        matches12 = np.stack([matches1, matches2], axis=1)
        db.add_matches(obj_name_to_image_ids[image_name1], obj_name_to_image_ids[image_name2], matches12)

        for obj in (image_name1, image_name2):
            num_matches_per_obj[obj] += len(matches12)
            num_pairs_per_obj[obj] += 1

    db.commit()
    db.close()

    print('\nRaw matches per instance (before geometric verification):')
    for i in range(n):
        print('  instance {}: {} pairs, {} matches{}'.format(
            i, num_pairs_per_obj[i], num_matches_per_obj[i],
            ' <-- no features' if i in no_feature_obj else ''))

    pair_path = Path(pair_path)
    estimation_and_geometric_verification(database_path, pair_path)
    reconstruction = run_reconstruction(sfm_dir, database_path, images_dir, True, options=options)
    if reconstruction is None:
        raise RuntimeError(
            'COLMAP could not reconstruct any model. The pairwise matches are too weak -- check '
            'raw/temp/pairwise_match_viz/ and raw/temp/global_match_viz/, and see the '
            '"COLMAP registers only a subset of the instances" section of the README.')

    cameras, images, points3D = read_model(path=sfm_dir, ext=".bin")
    write_model(cameras, images, points3D, path=sfm_dir, ext=".txt")

    registered = sorted(int(image.name[0:3]) for image in images.values())
    missing = [i for i in range(n) if i not in registered]
    print('\nCOLMAP registered {}/{} instances: {}'.format(len(registered), n, registered))
    if missing:
        print('COLMAP could NOT register instances: {}'.format(missing))
        print('Stages 6-8 continue with the {} registered instances and write them to '
              'non_empty_indexes.txt. Train with --visible_num {} (or -1). To try to recover the '
              'missing ones, see the "COLMAP registers only a subset of the instances" section of '
              'the README.'.format(len(registered), len(registered)))

    poses_est = dump_poses(
        os.path.join(sfm_dir, "images.txt"),
        image_names,
        os.path.join(sfm_dir, "poses.npz")
    )
