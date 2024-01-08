import numpy as np
import torch

def angle_error_vec(v1, v2, degrees=False):
    n = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    error_rad = np.arccos(np.clip(np.sum(v1 * v2, axis=1) / (n ), -1.0, 1.0))
    if degrees:
        return np.rad2deg(error_rad)
    else:
        return error_rad

def rotation_distance(R1,R2,eps=1e-7, degrees=False):
    # http://www.boris-belousov.net/2016/12/01/quat-dist/
    R1 = R1[..., 0:3, 0:3]
    R2 = R2[..., 0:3, 0:3]
    LEN = len(R1.shape)
    if LEN == 2:
        assert np.allclose(np.linalg.det(R1), np.ones(1))
    elif LEN == 3:
        assert np.allclose(np.linalg.det(R1), np.ones(R1.shape[0]))
    else:
        raise ValueError
    
    R_diff = R1 @ R2.transpose(0,2,1)
    trace = R_diff[...,0,0]+R_diff[...,1,1]+R_diff[...,2,2]
    angle = np.arccos(np.clip((trace-1)/2, -1+eps,1-eps))  # numerical stability near -1/+1
    if degrees:
        return np.rad2deg(angle)
    else:
        return angle

def rotation_distance_numpy(R1,R2,eps=1e-7, degrees=False):
    # http://www.boris-belousov.net/2016/12/01/quat-dist/
    R1 = R1[..., 0:3, 0:3]
    R2 = R2[..., 0:3, 0:3]
    LEN = len(R1.shape)
    if LEN == 2:
        assert np.allclose(np.linalg.det(R1), np.ones(1))
    elif LEN == 3:
        assert np.allclose(np.linalg.det(R1), np.ones(R1.shape[0]))
    else:
        raise ValueError
    
    R_diff = R1 @ R2.transpose(0,2,1)
    trace = R_diff[...,0,0]+R_diff[...,1,1]+R_diff[...,2,2]
    angle = np.arccos(np.clip((trace-1)/2, -1+eps,1-eps))  # numerical stability near -1/+1
    if degrees:
        return np.rad2deg(angle)
    else:
        return angle

def rotation_distance_torch(R1,R2,eps=1e-7, degrees=False):
    # http://www.boris-belousov.net/2016/12/01/quat-dist/
    """
    [b,3,3], [b,3,3], tensor
    """
    R1 = R1[..., 0:3, 0:3]
    R2 = R2[..., 0:3, 0:3]
    LEN = len(R1.shape)
    if LEN == 2:
        assert torch.allclose(torch.linalg.det(R1), torch.ones(1).cuda())
        assert torch.allclose(torch.linalg.det(R2), torch.ones(1).cuda())
    elif LEN == 3:
        assert torch.allclose(torch.linalg.det(R1), torch.ones(R1.shape[0]).cuda())
        assert torch.allclose(torch.linalg.det(R2), torch.ones(R2.shape[0]).cuda())
    else:
        raise ValueError
    
    R_diff = R1 @ R2.transpose(2,1)
    trace = R_diff[...,0,0]+R_diff[...,1,1]+R_diff[...,2,2]
    angle = torch.arccos(torch.clip((trace-1)/2, -1+eps,1-eps))  # numerical stability near -1/+1
    if degrees:
        return torch.rad2deg(angle)
    else:
        return angle