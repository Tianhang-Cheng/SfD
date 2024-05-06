"""Dump orb-slam config yaml file"""
import os
import math
import argparse

from file_utils import get_intrinsics

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", type=str, required=True, help="Dataset root directory")
parser.add_argument("--split", type=str, required=True, help="Dataset split, one of [train|validation|test]")
parser.add_argument("--log_id", type=str, required=True, help="Log id")
parser.add_argument("--img_width", type=float, required=True, help="Image width")
parser.add_argument("--img_height", type=float, required=True, help="Image height")
parser.add_argument("--fps", type=int, default=None, help="Video FPS")
parser.add_argument("--output_fpath_override", type=str, default=None, help="output fpath to override the default")
parser.add_argument("--use_original_setting", action="store_true", default=False,
    help="Use the original setting from the paper with 60 deg FoV intrinsics and 15 fps")
args = parser.parse_args()

output_dir = os.path.join(args.dataset_root, "data", args.split, args.log_id)
assert(os.path.isdir(output_dir))
gt_fpath = os.path.join(args.dataset_root, args.split, f"{args.log_id}.txt")
# assert(os.path.isfile(gt_fpath))

fx, fy, cx, cy = get_intrinsics(gt_fpath, args.img_width, args.img_height, args.use_original_setting)
if args.use_original_setting:
    # note that this might not be true because GT might not be 15 fps for all videos, but this param
    # isn't used in ORB-SLAM so shouldn't matter
    fps = 15.0
    fname = "orb_slam_60fov.yaml"
else:
    # note that similarly our dumped videos are probably not 30 fps either
    fps = 30.0
    fname = "orb_slam.yaml"

if args.fps is not None:
    fps = args.fps

if args.output_fpath_override is not None:
    fname = args.output_fpath_override

with open(os.path.join(output_dir, fname), "w") as f:
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