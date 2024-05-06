"""Extract/check the timestamps and image names of the video snippet."""
import os
import sys
import argparse
import cv2
import numpy as np

from file_utils import get_intrinsics


####################################### Parse Args ################################################
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", type=str, required=True, help="Dataset root directory")
parser.add_argument("--log_id", type=str, required=True, help="Log id")
args = parser.parse_args()

# Symlink the image dir to a image_0_resized dir (called resized for legacy reasons, the images are not resized)
img_dir = os.path.join(args.dataset_root, "images", args.log_id)
log_dir = os.path.join(args.dataset_root, "run_data", args.log_id)
target_dir = os.path.join(log_dir, "image_0_resized")
assert(os.path.isdir(img_dir))
if not os.path.exists(target_dir):
    os.symlink(img_dir, target_dir)

# Create image_names.txt, times.txt
image_names = sorted(os.listdir(img_dir))
times = np.round(np.arange(len(image_names)) * float(1e6) / 15).astype(np.int) # 15 fps, microseconds
with open(os.path.join(log_dir, "image_names.txt"), "w") as f:
    for x in image_names:
        f.write("{}\n".format(x))

with open(os.path.join(log_dir, "times.txt"), "w") as f:
    for x in times:
        f.write("{}\n".format(x))


# Create intrinsics
img = cv2.imread(os.path.join(target_dir, image_names[0]))
img_height, img_width = img.shape[:2]
fx, fy, cx, cy = get_intrinsics("", img_width, img_height, True)
fps = 15.0

with open(os.path.join(log_dir, "orb_slam_60fov.yaml"), "w") as f:
    f.write("""%YAML:1.0

#--------------------------------------------------------------------------------------------
# Camera Parameters. Adjust them!
#--------------------------------------------------------------------------------------------

# Camera calibration and distortion parameters (OpenCV) 
Camera.fx: {}
Camera.fy: {}
Camera.cx: {}
Camera.cy: {}

Camera.k1: 0.0
Camera.k2: 0.0
Camera.p1: 0.0
Camera.p2: 0.0

# Camera frames per second 
Camera.fps: {}

# Color order of the images (0: BGR, 1: RGB. It is ignored if images are grayscale)
Camera.RGB: 1

#--------------------------------------------------------------------------------------------
# ORB Parameters
#--------------------------------------------------------------------------------------------

# ORB Extractor: Number of features per image
ORBextractor.nFeatures: 2000

# ORB Extractor: Scale factor between levels in the scale pyramid 	
ORBextractor.scaleFactor: 1.2

# ORB Extractor: Number of levels in the scale pyramid	
ORBextractor.nLevels: 8

# ORB Extractor: Fast threshold
# Image is divided in a grid. At each cell FAST are extracted imposing a minimum response.
# Firstly we impose iniThFAST. If no corners are detected we impose a lower value minThFAST
# You can lower these values if your images have low contrast			
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

#--------------------------------------------------------------------------------------------
# Viewer Parameters
#--------------------------------------------------------------------------------------------
Viewer.KeyFrameSize: 0.1
Viewer.KeyFrameLineWidth: 1
Viewer.GraphLineWidth: 1
Viewer.PointSize:2
Viewer.CameraSize: 0.15
Viewer.CameraLineWidth: 2
Viewer.ViewpointX: 0
Viewer.ViewpointY: -10
Viewer.ViewpointZ: -0.1
Viewer.ViewpointF: 2000\n\n""".format(fx, fy, cx, cy, fps))