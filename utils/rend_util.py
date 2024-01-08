import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import cv2
import torch
import imageio
import pdb

import numpy as np 
from torch.nn import functional as F
import matplotlib.pyplot as plt

def load_rgb(path):
    img = imageio.imread(path)[:, :, :3]
    img = np.float32(img)
    # turn ldr to hdr, and map to [0,1]
    if path.endswith('.exr'):
        raise ValueError
    img = img / 255.
    img = np.power(img, 2.2) 
    return img

def load_exr(path):
    if not path.endswith('.exr'):
        raise ValueError
    if not os.path.exists(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def load_mask(path):
    alpha = imageio.imread(path, as_gray=True)
    alpha = np.float32(alpha) / 255.
    object_mask = alpha > 0.5

    return object_mask

def set_elements_to_zero(tensor, x):
    # 创建一个与原始张量形状相同的新张量
    result_tensor = np.copy(tensor)

    # 将 x 中的值设为 0，用于后续标记需要置为0的元素
    for val in x:
        result_tensor[result_tensor == val] = 0

    return result_tensor

def load_seg(path,
            input_range,
            output_range,
            non_empty_indexes=None):
    """
    '0_n' means segmentation range is (0, 1, 2, ..., n), where n is the number of objects
    '0_255' means linearly scale the segmentation range to (0, 1/n, 2/n, ..., 1) * 255
    """
    assert input_range in ['0_255', '0_n']
    assert output_range in ['0_255', '0_n']

    seg = imageio.imread(path)  # [h,w]
    n = len(np.unique(seg)) - 1 # 0 is background, so -1

    if input_range == '0_255' and output_range == '0_n':
        assert seg.max() == 255, 'input range is 0_255, but max value is {}'.format(seg.max())
        seg = np.round(seg / 255. * n) # 0 ~ n
    elif input_range == '0_n' and output_range == '0_255':
        seg = np.round(seg / n * 255)

    if non_empty_indexes is None:
        return seg

    raise ValueError
    empty_indexes = np.array(list(set(range(0, same_obj_num)) - set(non_empty_indexes)))
    if len(empty_indexes) > 0:
        print('before:', np.unique(seg))
        seg = set_elements_to_zero(seg, empty_indexes + 1)
        left = np.unique(seg) 
        for i in range(len(left)-1):
            seg[seg == left[i+1]] = i + 1 
        print('after:', np.unique(seg)) 
    if seg.max() > same_obj_num:
        import pdb
        pdb.set_trace()
        raise ValueError 
    return seg

def load_depth(path):
    depth = imageio.imread(path)
    depth = np.float32(depth)
    return depth

def load_depth_uv(path):
    uv = np.load(path)
    return uv

def load_depth_sfm(path):
    depth = np.load(path)
    return depth

def load_normal(path):
    """
    value range: [0, 1]
    """
    normal = imageio.imread(path)
    normal = np.float32(normal) / 255.0
    return normal

def get_camera_params(uv, pose, intrinsics):
    """
    Here pose is camera to world.
    """
    if pose.shape[1] == 7: # In case of quaternion vector representation
        cam_loc = pose[:, 4:]
        R = quat_to_rot(pose[:,:4])
        p = torch.eye(4).repeat(pose.shape[0],1,1).cuda().float()
        p[:, :3, :3] = R
        p[:, :3, 3] = cam_loc
    else: # In case of pose matrix representation
        cam_loc = pose[:, :3, 3]
        p = torch.eye(4).repeat(pose.shape[0],1,1).cuda().float()
        p[:, :3, :4] = pose[:, :3, :4]

    batch_size, num_samples, _ = uv.shape
    assert batch_size == 1

    depth = torch.ones((batch_size, num_samples)).cuda()
    x_cam = uv[:, :, 0].view(batch_size, -1) # [1, h*w]
    y_cam = uv[:, :, 1].view(batch_size, -1)
    z_cam = depth.view(batch_size, -1)

    p_uv = torch.cat([x_cam, y_cam, z_cam], dim=0)
    p_camera = (torch.inverse(intrinsics)[0] @ p_uv)  # [3, h*w]
    p_camera = p_camera * torch.tensor([1, -1, -1]).reshape(3, 1).cuda() # to NeuS coordinate
    p_world = (torch.bmm(p[:,0:3,0:3], p_camera[None]) + p[:,0:3, 3:4]).permute(0, 2, 1) # [1, h*w, 3]
    ray_dirs = p_world - cam_loc[:, None, :]
    ray_dirs = F.normalize(ray_dirs, dim=2)
    return ray_dirs, cam_loc

def quat_to_rot(q):
    batch_size, _ = q.shape
    q = F.normalize(q, dim=1)
    R = torch.ones((batch_size, 3,3)).cuda()
    qr=q[:,0]
    qi = q[:, 1]
    qj = q[:, 2]
    qk = q[:, 3]
    R[:, 0, 0]=1-2 * (qj**2 + qk**2)
    R[:, 0, 1] = 2 * (qj *qi -qk*qr)
    R[:, 0, 2] = 2 * (qi * qk + qr * qj)
    R[:, 1, 0] = 2 * (qj * qi + qk * qr)
    R[:, 1, 1] = 1-2 * (qi**2 + qk**2)
    R[:, 1, 2] = 2*(qj*qk - qi*qr)
    R[:, 2, 0] = 2 * (qk * qi-qj * qr)
    R[:, 2, 1] = 2 * (qj*qk + qi*qr)
    R[:, 2, 2] = 1-2 * (qi**2 + qj**2)
    return R


def rot_to_quat(R):
    batch_size, _,_ = R.shape
    q = torch.ones((batch_size, 4)).cuda()

    R00 = R[:, 0,0]
    R01 = R[:, 0, 1]
    R02 = R[:, 0, 2]
    R10 = R[:, 1, 0]
    R11 = R[:, 1, 1]
    R12 = R[:, 1, 2]
    R20 = R[:, 2, 0]
    R21 = R[:, 2, 1]
    R22 = R[:, 2, 2]

    q[:,0]=torch.sqrt(1.0+R00+R11+R22)/2
    q[:, 1]=(R21-R12)/(4*q[:,0])
    q[:, 2] = (R02 - R20) / (4 * q[:, 0])
    q[:, 3] = (R10 - R01) / (4 * q[:, 0])
    return q


def get_sphere_intersection(cam_loc, ray_directions, r = 1.0):
    ''' Input: n_images x 4 x 4 ; n_images x n_rays x 3
        Output: n_images * n_rays x 2 (close and far) ; n_images * n_rays '''

    n_imgs, n_pix, _ = ray_directions.shape

    cam_loc = cam_loc.unsqueeze(-1)
    ray_cam_dot = torch.bmm(ray_directions, cam_loc).squeeze()
    under_sqrt = ray_cam_dot ** 2 - (cam_loc.norm(2,1) ** 2 - r ** 2)

    under_sqrt = under_sqrt.reshape(-1)
    mask_intersect = under_sqrt > 0

    sphere_intersections = torch.zeros(n_imgs * n_pix, 2).cuda().float()
    sphere_intersections[mask_intersect] = torch.sqrt(under_sqrt[mask_intersect]).unsqueeze(-1) * torch.Tensor([-1, 1]).cuda().float()
    sphere_intersections[mask_intersect] -= ray_cam_dot.reshape(-1)[mask_intersect].unsqueeze(-1)

    sphere_intersections = sphere_intersections.reshape(n_imgs, n_pix, 2)
    sphere_intersections = sphere_intersections.clamp_min(0.01)
    mask_intersect = mask_intersect.reshape(n_imgs, n_pix)

    return sphere_intersections, mask_intersect


def get_depth(points, pose, scale=1.0):
    ''' Retruns depth from 3D points according to camera pose '''
    batch_size, num_samples, _ = points.shape
    if pose.shape[1] == 7:  # In case of quaternion vector representation
        cam_loc = pose[:, 4:]
        R = quat_to_rot(pose[:, :4])
        pose = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).cuda().float()
        pose[:, :3, 3] = cam_loc
        pose[:, :3, :3] = R

    points_hom = torch.cat((points, torch.ones((batch_size, num_samples, 1)).cuda()), dim=2)

    # permute for batch matrix product
    points_hom = points_hom.permute(0, 2, 1)
    device = pose.device
    pose=pose.cpu()
    inv_pose=torch.inverse(pose).to(device) 
    points_cam = torch.bmm(inv_pose, points_hom) # [1,4,h*w]
    depth = torch.norm(points_cam[:, 0:3], p=2, dim=1)[:, :, None] * scale

    return depth

def w2c_normal(normals, pose):
    """
    pose: local to world
    inv pose: world to local

    normals [1,h*w,3]
    pose [1,4,4]
    """
    inv_pose = np.linalg.inv(pose[:,0:3,0:3].cpu().numpy())
    inv_pose = torch.from_numpy(inv_pose).cuda()
    normals_local = torch.bmm(inv_pose, normals.permute(0, 2, 1)) 

    return normals_local.permute(0, 2, 1)

def draw_p(x, c='blue', y=None):
    if isinstance(x, torch.Tensor):
        points = x.detach().cpu().numpy()
    else:
        points = x
    
    if y is not None:
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy() 
    
    if isinstance(c, torch.Tensor):
        c = c.detach().cpu().numpy()

    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points[...,0],
             points[...,1],
             points[...,2],
            cmap='Accent',
            # s=0.5,
            c=c,
            # linewidth=0,
            alpha=1,
            marker=".")
    
    if y is not None:
        ax.scatter(y[...,0],
                y[...,1],
                y[...,2],
                cmap='Accent',
                s=5,
                c='red',
                # linewidth=0,
                alpha=1,
                marker=".")

    plt.title('Point Cloud') 
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def draw_img(x):
    if isinstance(x, torch.Tensor):
        img = x.detach().cpu().numpy()
    else:
        img = x
    plt.imshow(img)
    plt.show()