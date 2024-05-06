"""Extract/check the timestamps and image names of the video snippet."""
import os
import sys
import argparse

from file_utils import read_gt_trajectory, read_timestamps, read_str_file, align_timestamps


####################################### Parse Args ################################################
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", type=str, required=True, help="Dataset root directory")
parser.add_argument("--split", type=str, required=True, choices=["train", "validation", "test"],
    help="Dataset split, one of [train|validation|test]")
parser.add_argument("--log_id", type=str, required=True, help="Log id")
parser.add_argument("--snippet_id", type=str, default=None, help="Snippet id for the video snippet")
parser.add_argument("--extract_from_gt", action="store_true", default=False,
    help="Extract the video snippet corresponding to the GT sequence")
parser.add_argument("--exact_alignment", action="store_true", default=False,
    help="Use the original setting from the dataset with a lower fps rate. "
         "Only invoked when --extract_from_gt is also set")
parser.add_argument("--no_buffer", action="store_true", default=False,
    help="Do not prepend 15 frames as buffer for ORB-SLAM initialization failure. "
         "Only invoked when --extract_from_gt is also set")
args = parser.parse_args()

# Load all video frame timestamps
data_dir = os.path.join(args.dataset_root, "data", args.split, args.log_id)
assert(os.path.isdir(data_dir))
times_all_fpath = os.path.join(data_dir, "times_all.txt")
times_all = read_timestamps(times_all_fpath)
snippet_dir = os.path.join(args.dataset_root, "snippet", args.split, args.log_id)


####################################### Helper Functions ##########################################
def extract_frame_ids_from_times(times_all, times_snippet, exact_matches=True):
    """Return indices in times_all that match entries in times_snippet.
    Input:
        times_all: list of all timestamps
        times_snippet: list of subset of times_all
        exact_matches: if True, return indices of times_all that exactly correspond
            to entries in times_snippet; if False, return [i:j] where times_all[i]
            matches times_snippet[0] and times_all[j] matches times_snippet[-1]
    Output:
        [idx: times_all[idx] == t for t in times_snippet]
    """
    etol = min(1000, max(0, (times_all[1] - times_all[0]) / 2))  # microseconds
    frame_ids = align_timestamps(times_all, times_snippet, etol)
    if exact_matches:
        return frame_ids
    else:
        if len(frame_ids) > 0:
            return list(range(frame_ids[0], frame_ids[-1]+1, 1))
        else:
            return frame_ids

def extract_frame_ids_from_images(img_names):
    frame_ids = []
    for name in img_names:
        # name should be %06d.png
        assert(len(name)==10 and name.endswith(".png"))
        frame_ids.append(int(name[:6]))
    return frame_ids

def write_times(frame_ids, output_fpath):
    with open(output_fpath, "w") as f:
        for i in frame_ids:
            f.write("{}\n".format(times_all[i]))

def write_image_names(frame_ids, output_fpath):
    with open(os.path.join(output_dir, "image_names.txt"), "w") as f:
        for i in frame_ids:
            f.write("{}\n".format("%06d.png"%i))


####################################### Processing Snippets #######################################
if args.extract_from_gt:
    output_dir = os.path.join(snippet_dir, "gt_orig_fps" if args.exact_alignment else "gt")
    os.makedirs(output_dir, exist_ok=True)
    # GT metadata fpath
    gt_fpath = os.path.join(args.dataset_root, args.split, f"{args.log_id}.txt")
    assert(os.path.isfile(gt_fpath))

    traj_gt = read_gt_trajectory(gt_fpath)
    times_gt = traj_gt.timestamps

    frame_ids = extract_frame_ids_from_times(times_all, times_gt, exact_matches=args.exact_alignment)
    if len(frame_ids) == 0:
        print(f"No frames in {times_all_fpath} matched the GT times. "
        "Please double check the timestamps.")
    else:
        # We prepend 15 frames as buffer for ORB-SLAM initialization
        if not args.no_buffer:
            frame_id_gap = frame_ids[1] - frame_ids[0]
            frame_ids = list(range(max(0, frame_ids[0] - 15 * frame_id_gap), frame_ids[0], frame_id_gap)) \
                + frame_ids
    
        # Dump the times and image names for the snippet
        write_times(frame_ids, os.path.join(output_dir, "times.txt"))
        write_image_names(frame_ids, os.path.join(output_dir, "image_names.txt"))
else:
    assert(args.snippet_id is not None)
    output_dir = os.path.join(snippet_dir, args.snippet_id)
    # output_dir should already exist, and at least one of "times.txt" and "image_names.txt" exists
    assert(os.path.isdir(output_dir))
    times_fpath = os.path.join(output_dir, "times.txt")
    images_fpath = os.path.join(output_dir, "image_names.txt")
    assert(os.path.isfile(times_fpath) or os.path.isfile(images_fpath))

    frame_ids_times = []
    frame_ids_imgs = []
    if os.path.isfile(times_fpath):
        times_snippet = read_timestamps(times_fpath)
        frame_ids_times = extract_frame_ids_from_times(times_all, times_snippet, exact_matches=True)
        if not os.path.isfile(images_fpath):
            write_image_names(frame_ids_times, images_fpath)
    if os.path.isfile(images_fpath):
        img_names_snippet = read_str_file(images_fpath)
        frame_ids_imgs = extract_frame_ids_from_images(img_names_snippet)
        if len(frame_ids_times) > 0:
            # Both files should correspond to the same frame ids
            assert(frame_ids_times == frame_ids_imgs)
            print(f"Times and image name files for the snippet are consistent for snippet at {output_dir}")
        else:
            write_times(frame_ids_imgs, times_fpath)