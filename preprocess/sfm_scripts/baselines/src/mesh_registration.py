import numpy as np
import open3d as o3d
import os
import copy


######################################## Constructors #############################################
def get_GT_correspondences(num_vertices):
    ind_lst = np.arange(num_vertices).reshape(-1, 1)
    correspondences = np.hstack([ind_lst, ind_lst])
    correspondences_o3d = o3d.utility.Vector2iVector(correspondences)
    return correspondences_o3d


def create_point_cloud(pc, mask):
    new_pc = o3d.geometry.PointCloud(
        points=o3d.utility.Vector3dVector(np.asarray(pc.points)[mask]))
    new_pc.normals = o3d.utility.Vector3dVector(np.asarray(pc.normals)[mask])
    return new_pc


def filter_point_cloud_by_body_parts(
    pc1, pc2, faces, smpl_face_to_dp_face_mapping,
    dp_face_to_dp_part_mapping, body_part_lst):
    """Find the sub-pointcloud corresponding to the body_part_lst"""
    assert(len(faces) == len(smpl_face_to_dp_face_mapping))
    dp_parts = dp_face_to_dp_part_mapping[smpl_face_to_dp_face_mapping]
    parts_mask = np.array([False] * len(faces))
    for part in body_part_lst:
        parts_mask |= (dp_parts==part)
    parts_vert_inds = list(set(list(faces[parts_mask].reshape(-1))))
    pc1_new = create_point_cloud(pc1, parts_vert_inds)
    pc2_new = create_point_cloud(pc2, parts_vert_inds)
    return pc1_new, pc2_new


######################################## Registration #############################################
def vanilla_ICP_P2L(pc1, pc2, relative_pose_init, threshold=0.05):
    reg_p2l = o3d.pipelines.registration.registration_icp(
        pc1, pc2, threshold, relative_pose_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    return reg_p2l.transformation


def vanilla_ICP_P2P(pc1, pc2, relative_pose_init, threshold=0.05):
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pc1, pc2, threshold, relative_pose_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000))
    return reg_p2p.transformation


def estimate_from_correspondences(pc1, pc2, correspondences):
    estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    pose = estimator.compute_transformation(pc1, pc2, correspondences)
    return pose


def ICP_from_correspondences(pc1, pc2, correspondences, threshold=0.05):
    relative_pose_init = estimate_from_correspondences(pc1, pc2, correspondences)
    return vanilla_ICP_P2L(pc1, pc2, relative_pose_init, threshold=threshold)


def ransac_from_correspondences(pc1, pc2, correspondences, threshold=0.05, ransac_n=3):
    reg_ransac_p2p = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        pc1, pc2, correspondences, threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        ransac_n=ransac_n
    )
    return reg_ransac_p2p.transformation


######################################## Evaluation ###############################################
def eval_reg(pc1, pc2, pose, threshold=0.05):
    reg_result = o3d.pipelines.registration.evaluate_registration(
        pc1, pc2, threshold, transformation=pose
    )
    print(reg_result)
    print()


def draw_registration_result(source, target, transformation=np.eye(4)):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


########################################### Wrapper ###############################################
def mesh_registration(
    vertices1, vertices2, faces,
    smpl_rotmat1, smpl_rotmat2, smpl_face_to_dp_face_mapping, dp_face_to_dp_part_mapping,
    registration_type, registration_thresh=0.05, body_part_lst=None, visualize=False):
    """Register two meshes.
    Input:
        vertices1, vertices2: N x 3 numpy array indicating the vertex positions of the two meshes.
        faces: F x 3 numpy array indicating the faces shared by both meshes.
        smpl_rotmat1, smpl_rotmat2: 24 x 3 x 3 rotations from smpl where the first matrix is the root rotation.
        smpl_face_to_dp_face_mapping: a list of size F mapping smpl face to dp face index.
        dp_face_to_dp_part_mapping: a list mapping dp face to the dp part (e.g. torso)
        registration_type: one of "vanilla_ICP_P2L", "vanilla_ICP_P2P", "direct", "ransac"
        registration_threshold: max correspondence distance used in registration
        body_part_lst: body parts to use for the registration. None means we use all parts
        visualize: whether or not to visualize the registration
    Output:
        4x4 matrix indicating the transform from mesh1 to mesh2.
    """
    mesh1 = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vertices1),
        triangles=o3d.utility.Vector3iVector(faces))

    mesh2 = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vertices2),
        triangles=o3d.utility.Vector3iVector(faces))

    rot1 = smpl_rotmat1[0]
    rot2 = smpl_rotmat2[0]
    smpl_root_init = np.eye(4)
    smpl_root_init[:3, :3] = np.matmul(rot2.transpose(), rot1)

    # convert the meshes to point clouds
    pc1_full = o3d.geometry.PointCloud(
        points=mesh1.vertices)
    mesh1.compute_vertex_normals()
    pc1_full.normals = mesh1.vertex_normals

    pc2_full = o3d.geometry.PointCloud(
        points=mesh2.vertices)
    mesh2.compute_vertex_normals()
    pc2_full.normals = mesh2.vertex_normals

    # mask the point clouds according to the body types specified
    if body_part_lst is not None:
        # 1, 2 = Torso, 3 = Right Hand, 4 = Left Hand, 5 = Left Foot, 6 = Right Foot,
        # 7, 9 = Upper Leg Right, 8, 10 = Upper Leg Left, 11, 13 = Lower Leg Right, 
        # 12, 14 = Lower Leg Left, 15, 17 = Upper Arm Left, 16, 18 = Upper Arm Right, 
        # 19, 21 = Lower Arm Left, 20, 22 = Lower Arm Right, 23, 24 = Head
        pc1, pc2 = filter_point_cloud_by_body_parts(
            pc1_full, pc2_full, faces, smpl_face_to_dp_face_mapping, dp_face_to_dp_part_mapping,
            body_part_lst)
    else:
        pc1, pc2 = pc1_full, pc2_full
    
    gt_correspondences = get_GT_correspondences(len(np.asarray(pc1.points)))

    if registration_type == "vanilla_ICP_P2L":
        pose = vanilla_ICP_P2L(pc1, pc2, smpl_root_init, threshold=registration_thresh)
    elif registration_type == "vanilla_ICP_P2P":
        pose = vanilla_ICP_P2P(pc1, pc2, smpl_root_init, threshold=registration_thresh)
    elif registration_type == "direct":
        pose = estimate_from_correspondences(pc1, pc2, gt_correspondences)
    elif registration_type == "ransac":
        pose = ransac_from_correspondences(
            pc1, pc2, gt_correspondences, threshold=registration_thresh, ransac_n=200)
    else:
        raise NotImplementedError(f"Unknown registration method {registration_type}")

    if visualize:
        draw_registration_result(mesh1, mesh2, transformation=pose)
    
    eval_reg(pc1_full, pc2_full, pose, threshold=registration_thresh)
    return pose


def test():
    meshes = np.load("baselines/data/joyce_mesh.pkl", allow_pickle=True)
    noisy_meshes = np.load("baselines/data/joyce_mesh_2.pkl", allow_pickle=True)
    mapping = np.load("baselines/data/joyce_mapping.pkl", allow_pickle=True)

    REGISTRATION_THRESH = 0.05

    print("Full body vanilla P2L ICP")
    mesh_registration(
        noisy_meshes["verts1"], noisy_meshes["verts2"], meshes["faces"],
        noisy_meshes["rot_mat1"], noisy_meshes["rot_mat2"], mapping['dp_face_to_dp_part'],
        mapping['smpl_face_to_dp_face'], "vanilla_ICP_P2L", registration_thresh=REGISTRATION_THRESH, body_part_lst=None)
    
    print("Torso vanilla P2L ICP")
    mesh_registration(
        noisy_meshes["verts1"], noisy_meshes["verts2"], meshes["faces"],
        noisy_meshes["rot_mat1"], noisy_meshes["rot_mat2"], mapping["smpl_face_to_dp_face"],
        mapping["dp_face_to_dp_part"], "vanilla_ICP_P2L", registration_thresh=REGISTRATION_THRESH, body_part_lst=[1, 2])
    
    print("Torso+Head+UpperLeg vanilla P2L ICP")
    mesh_registration(
        noisy_meshes["verts1"], noisy_meshes["verts2"], meshes["faces"],
        noisy_meshes["rot_mat1"], noisy_meshes["rot_mat2"], mapping["smpl_face_to_dp_face"],
        mapping["dp_face_to_dp_part"], "vanilla_ICP_P2L", registration_thresh=REGISTRATION_THRESH, body_part_lst=[1, 2, 23, 24, 7, 8, 9, 10])
    
    print("Full body direct estimation")
    mesh_registration(
        noisy_meshes["verts1"], noisy_meshes["verts2"], meshes["faces"],
        noisy_meshes["rot_mat1"], noisy_meshes["rot_mat2"], mapping['dp_face_to_dp_part'],
        mapping['smpl_face_to_dp_face'], "direct", registration_thresh=REGISTRATION_THRESH, body_part_lst=None)
    
    print("Torso direct estimation")
    mesh_registration(
        noisy_meshes["verts1"], noisy_meshes["verts2"], meshes["faces"],
        noisy_meshes["rot_mat1"], noisy_meshes["rot_mat2"], mapping["smpl_face_to_dp_face"],
        mapping["dp_face_to_dp_part"], "direct", registration_thresh=REGISTRATION_THRESH, body_part_lst=[1, 2])
    
    print("Torso+Head+UpperLeg direct estimation")
    mesh_registration(
        noisy_meshes["verts1"], noisy_meshes["verts2"], meshes["faces"],
        noisy_meshes["rot_mat1"], noisy_meshes["rot_mat2"], mapping["smpl_face_to_dp_face"],
        mapping["dp_face_to_dp_part"], "direct", registration_thresh=REGISTRATION_THRESH, body_part_lst=[1, 2, 23, 24, 7, 8, 9, 10])
    
    print("Full body ransac with correspondences")
    mesh_registration(
        noisy_meshes["verts1"], noisy_meshes["verts2"], meshes["faces"],
        noisy_meshes["rot_mat1"], noisy_meshes["rot_mat2"], mapping['dp_face_to_dp_part'],
        mapping['smpl_face_to_dp_face'], "ransac", registration_thresh=REGISTRATION_THRESH, body_part_lst=None)
    
    print("Torso ransac with correspondences")
    mesh_registration(
        noisy_meshes["verts1"], noisy_meshes["verts2"], meshes["faces"],
        noisy_meshes["rot_mat1"], noisy_meshes["rot_mat2"], mapping["smpl_face_to_dp_face"],
        mapping["dp_face_to_dp_part"], "ransac", registration_thresh=REGISTRATION_THRESH, body_part_lst=[1, 2])
    
    print("Torso+Head+UpperLeg ransac with correspondences")
    mesh_registration(
        noisy_meshes["verts1"], noisy_meshes["verts2"], meshes["faces"],
        noisy_meshes["rot_mat1"], noisy_meshes["rot_mat2"], mapping["smpl_face_to_dp_face"],
        mapping["dp_face_to_dp_part"], "ransac", registration_thresh=REGISTRATION_THRESH, body_part_lst=[1, 2, 23, 24, 7, 8, 9, 10])


if __name__ == "__main__":
    test()