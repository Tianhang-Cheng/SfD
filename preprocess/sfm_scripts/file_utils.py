import os
import numpy as np
import csv
import cv2
import math

from evo.core.trajectory import PoseTrajectory3D
from scipy.spatial.transform import Rotation

from sfm_scripts.geo_utils import qvec2rotmat
from sfm_scripts.colmap_utils.database import COLMAPDatabase


def csv_read_matrix(file_path, delim=',', comment_str="#"):
    assert(os.path.isfile(file_path))
    with open(file_path, "r") as f:
        generator = (line for line in f
                        if not line.startswith(comment_str))
        reader = csv.reader(generator, delimiter=delim)
        mat = [row for row in reader]
    return mat


def read_gt_intrinsics(file_path):
    with open(file_path, "r") as f:
        line = list(map(float, f.readlines()[1].rstrip().split(" ")))
        assert(len(line) == 19)
        fx, fy, cx, cy = line[1:5]
    return fx, fy, cx, cy


def read_timestamps(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        times = list(map(float, [line.rstrip() for line in lines]))
    return times


def read_str_file(file_path):
    """Read a file made up of strings."""
    with open(file_path, "r") as f:
        lines = f.readlines()
    return [line.rstrip() for line in lines]


def read_gt_trajectory(file_path):
    """
    parses trajectory from the GT metadata txt file
    :param file_path: the metadata file path
    :return: trajectory.PoseTrajectory3D object
    """
    raw_mat = csv_read_matrix(file_path, delim=" ")
    error_msg = ("GT trajectory files must have 19 entries per row "
                 "and no trailing delimiter at the end of the rows (space)")
    raw_mat = csv_read_matrix(file_path, delim=" ")[1:]  # the first line is the youtube address
    if len(raw_mat) > 0 and len(raw_mat[0]) != 19:
        raise ValueError(error_msg)
    try:
        mat = np.array(raw_mat).astype(float)
    except ValueError:
        raise ValueError(error_msg)

    times = []
    poses = []
    for entry in mat:
        times.append(float(entry[0]))
        pose = np.eye(4)
        pose[:3, :] = entry[7:].reshape(3, 4)
        poses.append(np.linalg.inv(pose))  # gt pose is from world to camera frame, we need to invert it
    return PoseTrajectory3D(timestamps=times, poses_se3=poses)


def read_tum_trajectory(file_path):
    """
    parses trajectory file in TUM format (timestamp tx ty tz qx qy qz qw)
    :param file_path: the trajectory file path
    :return: trajectory.PoseTrajectory3D object
    """
    raw_mat = csv_read_matrix(file_path, delim=" ", comment_str="#")
    error_msg = ("TUM trajectory files must have 8 entries per row "
                 "and no trailing delimiter at the end of the rows (space)")
    if len(raw_mat) > 0 and len(raw_mat[0]) != 8:
        raise ValueError(error_msg)
    try:
        mat = np.array(raw_mat).astype(float)
    except ValueError:
        raise ValueError(error_msg)
    stamps = mat[:, 0]  # n x 1
    xyz = mat[:, 1:4]  # n x 3
    quat = mat[:, 4:]  # n x 4
    quat = np.roll(quat, 1, axis=1)  # shift 1 column -> w in front column
    return PoseTrajectory3D(xyz, quat, stamps)


def read_raw_colmap_trajectory(colmap_output_fpath, times_fpath, images_fpath, return_images=False, return_extrinsics=False):
    """
    parses trajectory in the raw COLMAP images.txt format
    :param colmap_output_fpath: the raw refined/images.txt trajectory file path
    :param times_fpath: timestamps of the trajectory
    :param images_fpath: image names of the trajectory
    :return: trajectory.PoseTrajectory3D object
    """
    times = read_timestamps(times_fpath)
    image_names = read_str_file(images_fpath)
    colmap_traj_image_names = []
    poses = []
    with open(colmap_output_fpath, "r") as f:
        lines = f.readlines()
        # first row of images.txt is a comment, poses are every other line
        # starting from the second row
        for i, line_idx in enumerate(range(1, len(lines), 2)):
            content = lines[line_idx].rstrip().split(" ")
            assert(len(content) == 10)
            colmap_traj_image_names.append(content[-1])
            pose = np.eye(4)
            qvec = list(map(float, content[1:5]))
            pose[:3, :3] = qvec2rotmat(qvec)
            pose[:3, 3] = list(map(float, content[5:8])) # pose is from world to camera frame
            if return_extrinsics:
                poses.append(pose)
            else:
                poses.append(np.linalg.inv(pose))
    # Reorder the poses according to ascending order of image names
    inds = np.argsort(colmap_traj_image_names)
    # assert(len(times) == len(poses))
    # assert(np.all(np.array(image_names) == np.array(colmap_traj_image_names)[inds]))
    times_inds = []
    time_idx = 0

    for idx in inds:
        while image_names[time_idx] != colmap_traj_image_names[idx]:
            time_idx += 1
        times_inds.append(time_idx)
    assert len(times_inds) == len(poses)
    if return_images:
        return PoseTrajectory3D(timestamps=np.array(times)[times_inds], poses_se3=np.array(poses)[inds]), image_names
    else:
        return PoseTrajectory3D(timestamps=np.array(times)[times_inds], poses_se3=np.array(poses)[inds])


def read_reality_capture_trajectory(poses_output_fpath, times_fpath, images_fpath, img_dir, return_extrinsics=False, convert_to_colmap=True, transpose=False):
    """
    parses trajectory in the raw COLMAP images.txt format
    :param poses_output_fpath: the raw poses.csv reality capture output file
    :param times_fpath: timestamps of the trajectory
    :param images_fpath: image names of the trajectory
    :return: trajectory.PoseTrajectory3D object
    """
    # times = read_timestamps(times_fpath)
    # image_names = read_str_file(images_fpath)
    traj_image_names = []
    poses = []
    intrinsics = []
    with open(poses_output_fpath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            traj_image_names.append(row['#name'])
            angles = np.deg2rad(np.array(list(map(float, [row['heading'], row['pitch'], row['roll']]))))
            pose = np.eye(4)
            R_rpy = Rotation.from_euler('ZYX', angles).as_matrix()
            R_rpy[:, 0] *= -1
            R_rpy[2, :] *= -1
            R_rpy[:, [0, 1]] = R_rpy[:, [1, 0]]
            R_rpy[[0, 1], :] = R_rpy[[1, 0], :]
            pose[:3, :3] = R_rpy
            # pose[:3, :3] = Rotation.from_euler('ZXZ', angles).as_matrix()
            # pose[1, :3] *= -1
            # pose[2, :3] *= -1
            # pose = euler_matrix(*angles, axes='rzxz')
            pose[:3, 3] = list(map(float, [row['x'], row['y'], row['alt']]))

            # convert to colmap format
            if convert_to_colmap:
                # convert to colmap format
                transform = np.array([
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ])
                pose = np.matmul(transform, pose)
            
            # intrinsics
            # see https://support.capturingreality.com/hc/en-us/articles/360017783459-RealityCapture-XMP-Camera-Math
            img_raw = cv2.imread(os.path.join(img_dir, row['#name']))
            im_h, im_w = img_raw.shape[0], img_raw.shape[1]
            im_h_mm, im_w_mm = 24, 36 # milli-meter for 35mm sensor.
            f_mm, px_mm, py_mm = map(float, [row['f'], row['px'], row['py']]) # millimeter
            f_px = f_mm/im_w_mm*im_w
            cx = px_mm*max(im_w, im_h) + im_w / 2.0
            cy = py_mm*max(im_w, im_h) + im_h / 2.0
            K = np.array([[f_px, 0, cx], [0, f_px, cy], [0, 0, 1]])

            if transpose:
                pose[:3, :3] = np.matmul(
                    pose[:3, :3],
                    np.array([
                        [0.0, 1.0, 0.0],
                        [-1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ]))
                K[[0, 1], 2] = K[[1, 0], 2]

            if return_extrinsics:
                pose = np.linalg.inv(pose)

            poses.append(pose)
            intrinsics.append(K)
    # Reorder the poses according to ascending order of image names
    # inds = np.argsort(traj_image_names)
    # times_inds = []
    # time_idx = 0
    # for idx in inds:
    #     while image_names[time_idx] != traj_image_names[idx]:
    #         time_idx += 1
    #     times_inds.append(time_idx)
    # return PoseTrajectory3D(timestamps=np.array(times)[times_inds], poses_se3=np.array(poses)[inds]), np.array(intrinsics)[inds], np.asarray(traj_image_names)[inds]
    return PoseTrajectory3D(timestamps=np.arange(len(traj_image_names)).astype(np.float), poses_se3=poses), intrinsics, traj_image_names


def read_reality_capture_trajectory_from_proj_mat(poses_output_dir, times_fpath, images_fpath, return_extrinsics=False, convert_to_colmap=True, transpose=False):
    def KRT_from_P(P):
        M = P[0:3,0:3]
        # QR decomposition
        q, r = np.linalg.qr(np.linalg.inv(M))
        R = np.linalg.inv(q)
        K = np.linalg.inv(r)
        # translation vector
        t = np.dot(np.linalg.inv(K),P[:,-1])
        D = np.array([[np.sign(K[0,0]),0,0],
                [0,np.sign(K[1,1]),0],
                [0,0,np.sign(K[2,2])]])
        # K,R,t correction
        K = np.dot(K, D)
        R = np.dot(np.linalg.inv(D), R)
        t = np.dot(np.linalg.inv(D), t)    
        t = np.expand_dims(t,axis=1)
        # normalize K
        K = K / K[-1,-1]
        return K, R, t
    
    # times = read_timestamps(times_fpath)
    # image_names = read_str_file(images_fpath)
    image_names = sorted([img_name for img_name in os.listdir(poses_output_dir) if img_name.endswith(".png")])
    times = np.arange(len(image_names)).astype(np.float)

    traj_times = []
    traj_img_names = []
    poses = []
    intrinsics = []
    proj_mats = []

    for time, image_name in zip(times, image_names):
        pose_fpath = os.path.join(poses_output_dir, f"{image_name[:-4]}_P.txt")
        if not os.path.isfile(pose_fpath):
            continue
        traj_times.append(time)
        traj_img_names.append(image_name)
        P = np.loadtxt(pose_fpath) # Projection matrix format
        K, R, t = KRT_from_P(P)
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t.reshape(-1)

        pose = np.linalg.inv(pose)  # cam to world
        # convert to colmap format
        if convert_to_colmap:
            # convert to colmap format
            transform = np.array([
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ])
            pose = np.matmul(transform, pose)
        
        if transpose:
            pose[:3, :3] = np.matmul(
                pose[:3, :3],
                np.array([
                    [0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]))
            K[[0, 1], 2] = K[[1, 0], 2]

        if return_extrinsics:
            pose = np.linalg.inv(pose)
        poses.append(pose)
        intrinsics.append(K)
        proj_mats.append(P)
    return PoseTrajectory3D(timestamps=traj_times, poses_se3=poses), intrinsics, traj_img_names, proj_mats


def read_raw_colmap_sparse_pointcloud(fpath):
    """
    parses 3d point cloud in the raw COLMAP pointsDD.txt format
    :param fpath: the raw points3D.txt file path
    :return: [Nx3] xyz positions, [Nx3] RGB colors normalized to [0, 1]
    """
    assert(os.path.isfile(fpath))
    xyz_pos = []
    rgb = []
    with open(fpath, "r") as f:
        lines = f.readlines()
        for point_info in lines:
            if point_info.startswith("#"):
                continue
            # 3D point list with one line of data per point:
            #   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
            point_info = list(map(float, point_info.rstrip().split(" ")))
            assert(len(point_info) > 8)
            xyz_pos.append([point_info[1], point_info[2], point_info[3]])
            rgb.append([point_info[4], point_info[5], point_info[6]])
    return np.array(xyz_pos), np.array(rgb) / 255.0


def read_raw_colmap_camera(fpath):
    """
    parses camera intrinsics (only one camera) from raw COLMAP cameras.txt file
    """
    assert os.path.isfile(fpath)
    with open(fpath, "r") as f:
        lines = f.readlines()
        assert len(lines) == 2
        content = lines[1].rstrip().split(" ")
        assert len(content) == 8
        assert content[1] == "PINHOLE"
        K = np.eye(3)
        fx, fy, cx, cy = list(map(float, content[4:]))
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return K


def get_colmap_image_name_to_id_mapping(database_fpath):
    """Return the image name to image id mapping from a COLMAP database.
    Input:
        database_fpath: path to the colmap database.db object
    Ouptut:
        {(image_name: image_id)} dictionary
    """
    db = COLMAPDatabase.connect(database_fpath)
    image_name_to_id = dict(
        (name, image_id)
        for image_id, name in db.execute(
            "SELECT image_id, name FROM images"))
    db.close()
    return image_name_to_id


def align_timestamps(times_all, times_subset, error_tolerance):
    """Find the indices in times_all where times_subset correspond to.
    Input:
        times_all: list of all timestamps
        times_subset: list of subset of times_all
        error_tolerance: error tolerance for ==
    Output:
        [idx: times_all[idx] == t +- error_tolerance for t in times_snippet]
    """
    i = 0
    j = 0
    frame_ids = []
    while i < len(times_all):
        if abs(times_all[i] - times_subset[j]) < error_tolerance:
            frame_ids.append(i)
            j += 1
            if j == len(times_subset):
                break
        i += 1
    return frame_ids


def get_intrinsics(gt_fpath, img_width, img_height, use_original_setting):
    """Extract camera intrinsics of a pinhole camera.
    Input:
        gt_fpath: path to the GT metadata file
        img_width: width of the camera image
        img_height: height of the camera image
        use_original_setting: if True, use the 60 deg FoV setting mentioned
            in the paper; if false, we use the intrinsics provided in the GT file.
    Output:
        fx, fy, cx, cy in the camera intrinsics matrix
        [[fx 0 cx], [0 fy cy], [0 0 1]]
    """
    if not use_original_setting:
        # Use the first GT intrinsics provided
        fx, fy, cx, cy = read_gt_intrinsics(gt_fpath)
        fx = fx * img_width
        cx = cx * img_width
        fy = fy * img_height
        cy = cy * img_height
    else:
        fov = math.pi/3  # 60 degrees
        fx = 1.0/math.tan(fov/2) * (img_width/2)
        fy = fx
        cx = 0.5 * img_width
        cy = 0.5 * img_height
    return fx, fy, cx, cy