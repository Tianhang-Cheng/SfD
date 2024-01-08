import numpy as np
import torch

from scipy.spatial.transform import Rotation

def get_eular_from_matrix_numpy(M, format="xyz"):
    """
    M [b,4,4]

    return [b,3]
    """ 
    rot = M[...,0:3,0:3] 
    r = Rotation.from_matrix(rot) 
    angles = r.as_euler(format, degrees=False) 
    return angles

def get_rotvec_from_matrix_numpy(M):
    rot = M[...,0:3,0:3] 
    r = Rotation.from_matrix(rot) 
    vec = r.as_rotvec()
    return vec  
def get_matrix_from_euler_numpy(euler, seq="xyz"):
    r = Rotation.from_euler(seq=seq, angles=euler, degrees=False)
    return r.as_matrix()

def get_matrix_from_rotvec_numpy(rotvec):
    r = Rotation.from_rotvec(rotvec)
    return r.as_matrix()

def to_square(M):
    """
    M [b,3,4]
    """
    if len(M.shape) == 3:
        homo = np.array([0,0,0,1]).reshape(1,1,4)
        homo = np.repeat(homo, repeats=M.shape[0], axis=0)
        M = np.concatenate([M, homo], axis=1)
    elif len(M.shape) == 2:
        homo = np.array([0,0,0,1]).reshape(1,4)
        M = np.concatenate([M, homo], axis=0)
    else:
        raise ValueError
    return M

def blender_to_opencv(pose):
    is_torch = False
    if isinstance(pose, torch.Tensor):
        pose = pose.numpy()
        is_torch = True
    flag = False
    if len(pose.shape) == 2:
        pose = pose[None]
        flag = True
    R_flip = get_matrix_from_rotvec_numpy(get_rotvec_from_matrix_numpy(pose) * np.array([[1,-1,-1]]))
    t_flip = (pose[:,0:3,3] * np.array([[1,-1,-1]]))[..., np.newaxis]
    RT_flip = np.concatenate([R_flip, t_flip], axis=2)
    RT_flip = to_square(RT_flip)
    if flag:
        RT_flip = RT_flip[0]
    if is_torch:
        RT_flip = torch.from_numpy(RT_flip)
    return RT_flip

def check_rotation_scale(M):
    if len(M.shape) == 3:
        assert np.allclose(np.linalg.det(M[:, 0:3, 0:3]), np.ones(M.shape[0])), 'det(R) != 1'
    elif len(M.shape) == 2:
        assert np.allclose(np.linalg.det(M[0:3, 0:3]), np.ones(M.shape[0])), 'det(R) != 1'