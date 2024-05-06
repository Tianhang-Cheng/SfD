import cv2
import open3d as o3d
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import csv
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
P = np.loadtxt('/content/girl_boxing/frame000001_P.txt')
K, R, t = KRT_from_P(P)
dist_coeff = np.array([-0.260165026, 0.100034012, 0, 0, -1.97E-02])
im = imread('/content/girl_boxing/frame000001.png')
im_h, im_w, im_c = im.shape
print(im_h, im_w, im_c)
pts = o3d.io.read_point_cloud('/content/girl_boxing/girl_boxing.ply')
x = np.array(pts.points).T
x_im = K.dot(R.dot(x) + t)
x_im = P.dot(np.concatenate((x, np.ones((1, x.shape[1])))))
z = x_im[2, :]
x = x_im[:2, :] / (x_im[2, :]+1e-6)
plt.figure(figsize=(20, 25))
plt.imshow(im)
plt.figure(figsize=(20, 25))
plt.imshow(np.ones_like(im))
idx = np.random.permutation(x.shape[1])[:int(x.shape[1]/2)]
plt.scatter(x[0, idx], x[1, idx], c = np.clip(np.asarray(pts.colors)[idx], 0, 100.0), s = 0.1)
plt.xlim([0, im_w])
plt.ylim([im_h, 0])
print(f'''K: {K}\n''', f'''R: {R}\n''', f'''t: {t}\n''')
with open('/content/girl_boxing/girl_boxing.csv', "r") as f:
  reader = csv.DictReader(f)
  row = next(reader)
angles = np.deg2rad(np.array(list(map(float, [row['heading'], row['pitch'], row['roll']]))))
pose = np.eye(4)
R_rpy = Rotation.from_euler('ZYX', angles).as_matrix()
# swaping x and y and flip in world, see the output mesh
R_cv2gl = np.array([
    [0.0, -1.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0]
])
R_euler = R_flip @ R_rpy.T @ R_cv2gl 
# flip z --> make sense, why do we flip camera coordinate between x and y? maybe related to the transposed image -- not sure.
R_flip = np.array([
    [0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0]
])
t_xya = -R_euler.dot(list(map(float, [row['x'], row['y'], row['alt']])))
print(f'''R: {R_euler}\n''', f'''t: {t_xya.T}\n''')
print(f'''R: {R}\n''', f'''t: {t}\n''')