import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True, help="Data directory containing the video")
parser.add_argument("--video_name", type=str, required=True, help="Video filename")
parser.add_argument("--fps", type=int, default=None, help="Desired FPS")
args = parser.parse_args()


vid_fpath = os.path.join(args.data_dir, args.video_name)
assert(os.path.isfile(vid_fpath))
img_dir = os.path.join(args.data_dir, "image_0")
os.makedirs(img_dir, exist_ok=True)

if args.fps is not None:
    fps_tolerance = 1000  # microseconds
    fps_interval = 1e6 / args.fps - fps_tolerance # microseconds
    print(f"Extracting images at {args.fps} FPS")
else:
    fps_interval = None

most_recent_time = None
with open(os.path.join(args.data_dir, "times_all.txt"), "w") as f:
    cap = cv2.VideoCapture(vid_fpath)
    frame_cnt = 0
    pose2time = {}        
    while True:
        grabbed, frame = cap.read()
        if not grabbed:
            break

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamp *= 1000 # convert to micro-seconds

        if most_recent_time is None or fps_interval is None or timestamp - most_recent_time > fps_interval:
            cv2.imwrite(os.path.join(img_dir, "%06d.png" % frame_cnt), frame)
            f.write("{}\n".format(timestamp))
            frame_cnt += 1
            most_recent_time = timestamp
    cap.release()
    print(f"There are {frame_cnt} frames in total.")