
# ***Structure from Duplicates**: Neural Inverse Graphics from a Pile of Objects*

[**Project Page**](https://tianhang-cheng.github.io/SfD-project.github.io/) | [**Paper**](https://tianhang-cheng.github.io/assets/pdf/dup_v3.pdf) | [**ArXiv**](https://arxiv.org/abs/2401.05236) | [**Full Dataset**](https://huggingface.co/datasets/TianhangCheng7/DuplicateSingleImage) | [**Blender Scenes**](https://huggingface.co/datasets/TianhangCheng7/DuplicateBlenderData) | [**Weights**](https://huggingface.co/TianhangCheng7/DuplicateWeight) | [**Results**](https://tianhang-cheng.github.io/SfD/)

## Preparation

Install pytorch 1.12 or higher version, the pytorch-lighting version can be found [**here**](https://lightning.ai/docs/pytorch/latest/versioning.html#pytorch-support)

```bash
conda create -n sfd python=3.9
conda activate sfd
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

# optional, if you want to use normal prior from pretrained model
pip install pytorch-lightning == 1.7.1 
```
Install other dependencies
```bash
pip install -r requirements.txt
```

### Download the pretrained weights and environment maps

This repository only tracks code. The large binaries live on the Hugging Face Hub and are pulled by
one script — run it once after cloning:

```bash
python download_assets.py
```

This fetches:

| what | where it lands | size | from |
| --- | --- | --- | --- |
| SuperPoint / SuperGlue checkpoints | `preprocess/keypoint_matching/weights/` | 92 MB | [TianhangCheng7/DuplicateWeight](https://huggingface.co/TianhangCheng7/DuplicateWeight) |
| `b`/`c`/`d` HDRI environment maps | `envmaps/*.exr` | 57 MB | [TianhangCheng7/DuplicateBlenderData](https://huggingface.co/datasets/TianhangCheng7/DuplicateBlenderData) |

You can also fetch one group at a time (`--weights`, `--envmaps`, `--force` to re-download). If you
forget to run the script, the code downloads what it needs on first use anyway — SuperPoint/SuperGlue
when the matcher is built, and an `.exr` when the `Mat` stage or `envmaps/fit_envmap_with_sg.py`
opens it.

Two more checkpoints are fetched by other libraries the first time they are needed, so the machine
that trains needs outbound network access once (or a warm cache):

| what | where it lands | size | from |
| --- | --- | --- | --- |
| torchvision VGG16 (the LPIPS backbone) | `~/.cache/torch/hub/checkpoints/` | 528 MB | [download.pytorch.org](https://download.pytorch.org/models/vgg16-397923af.pth) |

LPIPS is only used when validation images are plotted and in `--eval`, but it is computed at
iteration 0, so a machine that cannot reach `download.pytorch.org` fails early in training.

The Omnidata monocular-cue checkpoint is *not* included, see
[Data Preprocessing](#preprocessing-flow) below.

The sample dataset is included in /data
The model works in both Linux and Windows

## Data Preprocessing

Tips:
1. Instances that SfM cannot register are dropped automatically and you train on the rest — see
   [COLMAP registers only a subset of the instances](#colmap-registers-only-a-subset-of-the-instances).
2. The original image should have big enough resolution, otherwise there may not enough keypoints for SfM.

### Where to put your image

Create a new folder in `/data` for your input, e.g. `/data/your_object`, and a `raw` folder inside
it. Put your RGB image and instance segmentation in `/data/your_object/raw` and name them
`000_rgb.png` and `000_instance_seg.png`. (`train/` is an *output* of stage 0, do not create it
yourself.)

The folder structure will be:
```
/data
  /airplane
  /your_object
    /raw
      -000_rgb.png
      -000_instance_seg.png
```
Only `airplane` and `your_object` ship with this repository; every other object name mentioned in
the paper or in `datasets/data_info.py` has to be downloaded from
[DuplicateSingleImage](https://huggingface.co/datasets/TianhangCheng7/DuplicateSingleImage) or
built from your own image.

The instance seg can be obtained from Segment-anything (not provide here) or manual segmentation.
Its background should be 0, then the value of each instance area is 1/N×255, 2/N×255, 3/N×255, ..., N/N×255, where N is instance numbers.

### Preprocessing flow

0: crop each instance from the original image
1: find keypoints and match them for each pair
2-4: turn pair-wise matching to global matching
5: sfm
6-7: visualize and dump pose
8: dump surface normal from pretrained network, will be skipped if failed

For 5_sfm, please install [colmap](https://github.com/colmap/pycolmap) by 'pip install pycolmap==0.6.1'.
Newer wheels work too — the pipeline is tested on pycolmap 0.6.1 and 4.1.1.

**Input format note.** `000_rgb.png` must be an 8-bit image. If you feed a rendered `000_rgb.exr`
instead, stage 0 reads it with OpenCV (`IMREAD_ANYDEPTH`) and tonemaps it; do **not** read EXR with
`imageio`, which silently clips HDR data to `uint8` in `{0, 1}` and turns every instance crop into a
flat silhouette that SuperGlue cannot match. Stage 0 prints a loud warning if any instance crop ends
up with fewer than 16 distinct intensity levels.

### COLMAP registers only a subset of the instances

This is normal and the pipeline is designed to keep going. Stage 5 picks the **largest** model
COLMAP produced and prints, for example:

```
COLMAP registered 5/6 instances: [0, 1, 3, 4, 5]
COLMAP could NOT register instances: [2]
```

Stage 6 writes the survivors to `non_empty_indexes.txt`, and stages 6/7 compact them: the *k*-th
registered instance becomes object index *k* in `object_pred_pose.json` and `points_world.npy`, and
`utils/rend_util.load_seg` relabels the instance segmentation the same way, dropping the
unregistered instances to background. So you train on the registered subset — pass
`--visible_num -1` (or any value `<=` the number of registered instances) and the unregistered
instances are simply excluded. Everything stays consistent even when the missing instance is not
the last one.

If too many instances drop out, the cause is almost always weak pairwise matching, not COLMAP:

- Look at `raw/temp/pairwise_match_viz/` and `raw/temp/global_match_viz/`. If the matches are
  visibly wrong, no COLMAP setting will help.
- Check the "Raw matches per instance" table stage 5 prints. An instance with 0 pairs was thrown
  away earlier: stage 2 found no rotation with meaningful correspondence, or stage 4 found fewer
  than 8 matches for every one of its pairs.
- Reduce `rotate_delta_angle` (e.g. 8 → 4) so stage 1 searches relative rotations more finely.
  Cost grows as `360 / rotate_delta_angle * instance_num ** 2`.
- Increase `crop_size` if instances are large in the image, and keep it comfortably larger than the
  biggest instance bounding box — stage 0 raises an error if an instance does not fit.
- Instances that are strongly occluded, very small in pixels, or seen from a nearly degenerate
  viewpoint may simply have no usable keypoints. Dropping them is the expected behaviour.

To exercise stages 5-7 without the SuperPoint/SuperGlue checkpoints (useful for debugging this
path), `preprocess/debug_synth_sfm.py` fabricates stage-4 outputs from a known synthetic object:

```bash
python preprocess/debug_synth_sfm.py --instance_dir /tmp/synth --instance_num 6 --drop_instance 2
python preprocess/run.py --instance_dir /tmp/synth --instance_num 6 --stages 5-7
```

`--drop_instance K` removes every pair involving instance K so COLMAP cannot register it;
add `--drop_instance_feats` to also simulate stage 4 finding no good pair for it at all.
`--from_poses DIR` reuses the virtual cameras of an already-preprocessed object instead of a
made-up camera rig, so the poses stages 5-7 recover can be compared against that reference.

For 8_extract_monocular_cues.py, you should download the weight from [Omnidata](https://github.com/EPFL-VILAB/omnidata) and put the pretrained normal prediction network "omnidata_dpt_normal_v2.ckpt" to /preprocess/omnidata/omnidata_tools/torch/pretrained_models. This is the one checkpoint `download_assets.py` cannot fetch for you, because we do not redistribute it. `/preprocess/omnidata` is a trimmed copy of the upstream repo that keeps only the modules this step imports — see [its README](preprocess/omnidata/README.md).

### Start processing

Put your image and its instance segmentation in `data/<object_name>/raw/`:

```
data/your_object/raw/000_rgb.png          # or 000_rgb.exr for HDR input
data/your_object/raw/000_instance_seg.png # label 0 = background, 1..N = the instances
```

Then run the whole pipeline (stages 0-8) with one command:

```bash
bash preprocess/run.sh data/your_object
# equivalently: python preprocess/run.py --instance_dir data/your_object
```

The input image and the number of instances are detected from the data on disk, so there is
nothing to edit. Override anything explicitly when you need to:

```bash
python preprocess/run.py --instance_dir data/my_pile --instance_num 7 \
    --crop_size 1000 --train_res 800 --rotate_delta_angle 4
```

`--stages` re-runs part of the pipeline without redoing the expensive matching, e.g. to iterate
on the SfM step only:

```bash
python preprocess/run.py --instance_dir data/my_pile --stages 5-7
```

Stage 8 (Omnidata normals) is optional: if its checkpoint is missing the script warns and keeps
going, and you then train without `--use_pretrain_normal`. `python preprocess/run.py --dry_run`
prints the per-stage commands without running them.

The training data appears in `data/your_object/` and the script prints the training command to
run next. Adding your object to `datasets/data_info.py` is optional — `exp_runner.py` falls back
to reading the instance count from `train/000_instance_seg.png` and infers `real_world` from
whether `train/000_rgb.exr` exists, and both can be forced with `--same_obj_num N` /
`--real_world` / `--no-real_world`.

### Download the full dataset from Hugging Face

The full `DuplicateSingleImage` dataset (pre-processed, ready to train) is hosted on
Hugging Face Hub at
[**TianhangCheng7/DuplicateSingleImage**](https://huggingface.co/datasets/TianhangCheng7/DuplicateSingleImage).

Install the client if you don't already have it:
```bash
pip install -U huggingface_hub
```

Download the whole dataset:
```bash
hf download TianhangCheng7/DuplicateSingleImage --repo-type dataset --local-dir /path/to/DuplicateSingleImage
```

You can also do this from Python:
```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="TianhangCheng7/DuplicateSingleImage",
    repo_type="dataset",
    local_dir="/path/to/DuplicateSingleImage",
)
```

Older versions of `huggingface_hub` expose the same CLI as `huggingface-cli download` instead of
`hf download`.

### Layout of the downloaded dataset

`local-dir` ends up with two top-level folders, **not** a flat list of object folders:

```
DuplicateSingleImage
  /train_split
    /airplane
      points_world.npy
      transforms_train.json
      object_pred_pose.json
      non_empty_indexes.txt
      object_scale_matrix.json
      /train
        000_rgb.png (or .exr for synthetic objects)
        000_instance_seg.png
        000_normal_pretrain.png
        ...
    /box
    /cash
    ... (one folder per object, matching the names in datasets/data_info.py)
  /eval_split
    /airplane
      transforms_test.json
      depth_sfm_bar_origin.png
      /train
        000_mask.png
    /box
      transforms_test.json
      /train
        000_diffuse.png       # only for synthetic objects (albedo GT)
        000_roughness.png     # only for synthetic objects
        000_mask.png
      /test_relight_b         # only for objects with relighting GT
      /test_relight_d
    ...
```

- **`train_split/<object>`** is a ready-to-train instance folder — point `--data_split_dir` at it
  directly (e.g. `--data_split_dir /path/to/DuplicateSingleImage/train_split/airplane`), no
  copying into `/data` required.
- **`eval_split/<object>`** holds the held-out ground truth used only for evaluation (see
  [Evaluation](#evaluation) below). For real-world objects this is just a `test_mask`; for
  synthetic objects it also carries GT `000_diffuse.png` / `000_roughness.png` (albedo/roughness)
  and, for some objects, `test_relight_b` / `test_relight_d` frames for the relighting eval.
  Training never reads `eval_split` — only `--eval` / `--eval_relight` need it, and only for the
  `000_diffuse.png` / `000_roughness.png` files (copied next to the training frame as shown in
  [Batch training & evaluation](#batch-training--evaluation-cmd_trainsh--cmd_evalsh) below).
- Every object folder name under `train_split`/`eval_split` should already have a matching entry
  in `datasets/data_info.py`'s `obj_info` dict; add one if you add a new object.

### Raw Blender scenes (optional)

`DuplicateSingleImage` ships the rendered images. If you want the **3D source files** — to re-render
the synthetic objects, change the lighting, or build new scenes — the Blender projects are on the Hub
at [**TianhangCheng7/DuplicateBlenderData**](https://huggingface.co/datasets/TianhangCheng7/DuplicateBlenderData)
(~770 MB):

```bash
python download_assets.py --blender-data blender_data
# or: hf download TianhangCheng7/DuplicateBlenderData --repo-type dataset --local-dir blender_data
```

```
blender_data
  /box, /cash, /cleaner, /clock, /coffee, /fire, /gitar, /sign, /tin
    <object>_clean.blend        # the scene
    /textures                   # the PBR maps it references
  /hdi
    a.exr ... f.exr, nv_box.hdr # HDRI environment maps used to light the renders
```

`hdi/{b,c,d}.exr` are the same environment maps that `download_assets.py` drops into `envmaps/`, so
you don't need this download just to run the relighting evaluation.

### Training on a single pre-packaged sample

If you want to train just one object from the dataset above (rather than the whole batch — see
below for that), you can point `--data_split_dir` straight at its `train_split` folder and skip
this step entirely. Copying into `/data` is only needed if you want to keep the object next to the
bundled samples (`data/airplane`, `data/your_object`) or you built the folder yourself via the
"Data Preprocessing" pipeline above:

1. Copy the object folder into `/data`:
   ```bash
   mkdir -p data/your_object
   cp -r /path/to/DuplicateSingleImage/train_split/your_object/* data/your_object/
   ```
2. Optionally add an entry for the object name to `datasets/data_info.py`:
   ```python
   'your_object' : [instance_num, is_synthetic, training_resolution],
   ```
   `is_synthetic` should be `True` for objects with `blender_object_gt_pose.json` /
   `blender_camera_gt_pose.json` at the top level (rendered data), and `False` for real-world
   captures (no blender GT pose files, RGB stored as `.png` instead of `.exr`). Most of the
   objects that ship with `DuplicateSingleImage` already have an entry in this file. Without an
   entry, `exp_runner.py` reads the instance count from `train/000_instance_seg.png` and infers
   `real_world` from whether `train/000_rgb.exr` exists; `--same_obj_num N` and
   `--real_world` / `--no-real_world` override either.
3. Run the 3 training stages as described below, pointing `--data_split_dir` at `./data/your_object`.

> **Note** the bundled `data/airplane` is a *reference* preprocessing output and is missing
> `points_world.npy` (a stage-6 product), so it cannot be trained as-is — training fails with
> `FileNotFoundError: data/airplane/points_world.npy`. Download `airplane` from
> `DuplicateSingleImage` (above), or re-run `preprocess/run.py` on your own image.

## Training

Take ```airplane``` as example, we train the network in 3 stages. The checkpoints will be generated under /exps.

### Stage 1: Train geometry network (~10 hour)
```bash
python exp_runner.py \
  --conf configs/default.yaml \
  --data_split_dir ./data/airplane \
  --expname airplane \
  --trainstage Geo \
  --use_pretrain_normal \
  --init_method SFM
```

### Stage 2: Train visibility network (~30 minutes)

```bash
python exp_runner.py \
  --conf configs/default.yaml \
  --data_split_dir ./data/airplane \
  --expname airplane \
  --trainstage Vis \
  --init_method SFM
```

### Stage 3: Train material network (~1 hour)

```bash
python exp_runner.py \
  --conf configs/default.yaml \
  --data_split_dir ./data/airplane \
  --expname airplane \
  --trainstage Mat \
  --init_method SFM
```

Note for command: 
+ **--is_continue** : load from previous checkpoint
+ **--use_pretrain_normal** : add normal constrain from [MonoSDF](https://github.com/autonomousvision/monosdf). Model performance may decrease when pretrained normal has 
bad quality.
+ **--debug**: forbid visualization and run experiment in low sample numbers.

## Evaluation

After the Material stage (Stage 3) has produced a checkpoint, you can evaluate the trained model
against held-out ground truth. This requires a `test` split for the object: `transforms_test.json`,
plus `_rgb` (renamed from `_color` as above), `_diffuse` and `_roughness` ground-truth files for
the test frame(s), which is normally produced by the full preprocessing pipeline.

Evaluate rgb / albedo / normal / roughness against the `test` split:
```bash
python exp_runner.py \
  --conf configs/default.yaml \
  --data_split_dir ./data/your_object \
  --expname your_object \
  --trainstage Mat \
  --init_method SFM \
  --is_continue \
  --eval
```
This loads the latest Material checkpoint and reports PSNR/SSIM/LPIPS for rgb and albedo, and
error metrics for normal and roughness, under `exps/Mat-your_object-eval/<timestamp>/evals_value/`
(numeric results) and `evals_image/` (rendered images).

Evaluate relighting against `test_relight_b`/`test_relight_d` (only for objects that ship these
folders, e.g. some of the synthetic `DuplicateSingleImage` objects). Relighting eval first needs a spherical
Gaussian fit of the target environment map — run this once per envmap (`b`/`d`) if
`envmaps/b/sg_128.npy` / `envmaps/d/sg_128.npy` don't already exist:
```bash
python envmaps/fit_envmap_with_sg.py --envmap_path envmaps/b.exr --num_sg 128
python envmaps/fit_envmap_with_sg.py --envmap_path envmaps/d.exr --num_sg 128
```
The `.exr` files are not in git; the script downloads the one it needs from
[TianhangCheng7/DuplicateBlenderData](https://huggingface.co/datasets/TianhangCheng7/DuplicateBlenderData)
if it is missing (or grab all of them up front with `python download_assets.py --envmaps`).
Then run:
```bash
python exp_runner.py \
  --conf configs/default.yaml \
  --data_split_dir ./data/your_object \
  --expname your_object \
  --trainstage Mat \
  --init_method SFM \
  --is_continue \
  --eval_relight
```
Results are written under `exps/Mat-your_object-eval-relight/<timestamp>/{b,d}/`.

Note: in earlier versions of this repo, both `--eval` and `--eval_relight` immediately raised
`NotImplementedError` in `exp_runner.py` even though the evaluation code itself
(`MaterialTrainRunner.evaluate()` / `evaluate_envmap()` / `evaluate_relight()`) was fully
implemented; those stubs have been removed so the flags above work as documented.

## Batch training & evaluation (cmd_train.sh / cmd_eval.sh)

The single-object commands above are convenient for one object, but the downloaded
`DuplicateSingleImage` dataset ships 15 objects. `cmd_train.sh` and `cmd_eval.sh` drive all of
them through the same 3 training stages + eval, sharded round-robin across multiple GPUs, without
needing to invoke `exp_runner.py` by hand for every object/stage/GPU combination.

Both scripts assume the dataset was downloaded with `hf download` (or `snapshot_download`) as
described in [Layout of the downloaded dataset](#layout-of-the-downloaded-dataset) above, i.e. a
`train_split/<object>` and `eval_split/<object>` folder per object — they read directly from
`train_split`/`eval_split`, so there is no need to copy objects into `/data` first.

### 1. Batch training — `cmd_train.sh`

For every object in `SAMPLES=(airplane box cake cash cheese cleaner clock coffee cola fire gitar
potato sign tin yogurt)`, this runs Stage 1 (Geo) → Stage 2 (Vis) → Stage 3 (Mat) in order,
`cd`-ing into `SfD` and calling `exp_runner.py` exactly as in the "Training" section above with
`--data_split_dir "$DATA_ROOT/$name"` for each stage. Objects are split round-robin across
`NUM_GPUS` GPUs (`CUDA_VISIBLE_DEVICES` is set per worker), so with the default `NUM_GPUS=4` each
GPU trains its own subset of ~4 objects, one after another, while the other GPUs run in parallel.
If a stage fails for an object (e.g. OOM), that object's remaining stages are skipped but every
other object/GPU keeps going — nothing else aborts.

Before running, the scripts default `SFD_DIR` to their own location and derive the other paths
relative to it, expecting a layout like:
```
/path/to
  /SfD                       # this repo checkout (contains cmd_train.sh)
    /train_logs              # created automatically, gitignored
  /DuplicateSingleImage
    /train_split
    /eval_split
```
If your layout differs, override any of them via environment variables instead of editing the
script:
```bash
SFD_DIR=/path/to/SfD                                   # repo checkout
DATA_ROOT=/path/to/DuplicateSingleImage/train_split     # where you downloaded train_split
LOG_DIR=/path/to/train_logs                             # per-object logs go here
NUM_GPUS=4                                              # adjust to your GPU count
```
`SAMPLES` (edited directly in the script) can be trimmed to a subset if you only want to train
some objects.

Run it (inside `tmux` so it survives a disconnect — with the defaults, `cmd_train.sh` itself
estimates ~4 samples/GPU * (~10h Geo + ~0.5h Vis + ~1h Mat) ≈ 46h total wall-clock, since all 4
GPUs train their shards in parallel):
```bash
tmux new -s train
bash /path/to/SfD/cmd_train.sh
# detach with Ctrl-b d, reattach later with: tmux attach -t train
```
Progress and errors for object `<name>` land in `$LOG_DIR/<name>.log`; checkpoints land under
`exps/Geo-<name>`, `exps/Vis-<name>`, `exps/Mat-<name>` as usual. Tail a log while it runs with
`tail -f $LOG_DIR/<name>.log`, or check which objects are done with
`grep -l "DONE (all 3 stages)" $LOG_DIR/*.log`.

### 2. Batch evaluation — `cmd_eval.sh`

Run this only after `cmd_train.sh` has produced a Stage-3 (Mat) checkpoint for the objects you
want to evaluate. For each object it:
1. **Merges eval ground truth** — copies `eval_split/<name>/train/000_diffuse.png` and
   `000_roughness.png` (only present for synthetic objects) into `train_split/<name>/train/` if
   not already there, since (as noted in [Evaluation](#evaluation) and the dataset layout section)
   this single-view setup evaluates the same frame in place and just needs the GT albedo/roughness
   sitting alongside the training image — it does not copy `000_mask.png` or
   `transforms_test.json`, since the current eval code path doesn't read either.
2. Calls `exp_runner.py --trainstage Mat --init_method SFM --is_continue --eval` against the
   latest Mat checkpoint, same as the manual eval command above.

It uses the same GPU-sharding / failure-isolation scheme and path defaults/overrides
(`SFD_DIR`, `DATA_ROOT`, `EVAL_DATA_ROOT`, `LOG_DIR`, `NUM_GPUS`) as `cmd_train.sh`. Run it:
```bash
tmux new -s eval
bash /path/to/SfD/cmd_eval.sh
# detach with Ctrl-b d, reattach later with: tmux attach -t eval
```
Per-object logs land in `$LOG_DIR/<name>_eval.log`; numeric results and rendered comparisons land
under `exps/Mat-<name>-eval/<timestamp>/evals_value/` and `evals_image/` as described in
[Evaluation](#evaluation). Note that neither script runs `--eval_relight` — for the relighting
metrics you still need to run that command by hand per object as documented above.

### 3. Building an HTML report — `build_report.py` / `build_html.py`

Once `cmd_train.sh` and `cmd_eval.sh` have both finished for the objects you care about,
`results/build_report.py` and `results/build_html.py` (outside the `SfD` checkout, under
`/mnt/task_runtime/results`) turn the scattered per-object `exps/Mat-<name>-eval/...` output into
one browsable report:

```bash
cd /mnt/task_runtime/results
python3 build_report.py   # collects metrics + images -> results.json, metrics_plot.png, training_time_plot.png
python3 build_html.py     # renders results.json -> index.html
```
Then open `results/index.html` in a browser.

- **`build_report.py`** iterates the hardcoded `SAMPLES` list, for each object finds the latest
  `exps/Mat-<name>-eval/<timestamp>/` run, reads its `evals_value/*.txt` metrics plus
  `env_map_mse.txt` / `run_time.txt`, and copies `evals_image/*.png` into `results/assets/<name>/`.
  It also parses `train_logs/<name>.log` for the Geo/Vis/Mat stage timestamps to compute training
  hours. Everything is written to `results.json`, and `metrics_plot.png` /
  `training_time_plot.png` (per-object bar charts and a stacked training-time-per-stage chart) are
  plotted from the same data.
- **`build_html.py`** reads `results.json` and renders a single self-contained
  `index.html`: a summary table (with a means row) across all objects, the two plot images, and a
  per-object gallery of GT-vs-Ours image pairs (rgb / albedo / normal / roughness / metallic).
- **Missing ground truth is shown as N/A, not a fabricated number.** Real-world objects
  (`is_synthetic=False` in `datasets/data_info.py`, i.e. `airplane`, `cake`, `cheese`, `cola`,
  `potato`, `yogurt`) have no albedo/roughness/normal ground truth — `datasets/neus_dataset.py`
  substitutes a blank placeholder image for all three instead of a real capture — and no model
  ever produces metallic ground truth (`metallic_gt=None` in `trainer/train_material.py`). For
  those metric/image slots, `build_report.py` writes `null` into `results.json` and replaces the
  gallery image with an explicit "N/A" placeholder graphic, so `build_html.py`'s table/gallery and
  `metrics_plot.png` render them as gaps (`—` / excluded from the mean) rather than a real-looking
  but meaningless value.
- Both scripts hardcode `ROOT = /mnt/task_runtime` and expect `SfD/exps` and `train_logs` as
  siblings under it (unlike `cmd_train.sh`/`cmd_eval.sh`, they don't read path overrides from the
  environment) — edit the `ROOT`/`SFD_DIR`/`LOG_DIR`/`OUT_DIR` constants near the top of
  `build_report.py` if your layout differs. Note this also means `LOG_DIR` defaults to
  `/mnt/task_runtime/train_logs`, not `cmd_train.sh`'s own default of `$SFD_DIR/train_logs` — if
  you didn't override `LOG_DIR` when running `cmd_train.sh`, either pass `LOG_DIR=$SFD_DIR/train_logs`
  to it next time or update the constant in `build_report.py` to match, otherwise train-time hours
  will show up empty in the report.

## TODO
**[√]** release training code\
**[√]** release sample data\
**[√]** release eval code\
**[√]** release full dataset\
**[√]** release pre-process code\
**[ ]** release pretrained weight\
**[ ]** extract mesh and texture from network

## Others

### Coordinate System

<img src="description/coord.PNG" width = "80%" />

### OOM
You can decrease ```geo_num_pixels```, ```vis_num_pixels``` or ```mat_num_pixels``` if out of memory

### Training Visualization

#### Input 

<img src="description/input_airplane.png" width = "61%" border=0>

Image | Instance mask

#### Geometry Stage

<table><tr>
<td><img src="description/rgb_airplane.gif" width = "100%" border=0></td>
<td><img src="description/nrm_airplane.gif" width = "100%" border=0></td>
<td><img src="description/error_airplane.gif" width = "100%" border=0></td>
</tr></table>

Appearence (500iter/frame) | Surface Normal (500iter/frame) | Rendering Error (500iter/frame)

#### Material Stage

<table><tr>
<td><img src="description/dif_airplane.gif" width = "100%" border=0></td>
<td><img src="description/rough_airplane.gif" width = "100%" border=0></td>
<td><img src="description/rerender_airplane.gif" width = "100%" border=0></td>
</tr></table>

Diffuse (1000iter/frame) | Roughness (1000iter/frame) | Rerender (1000iter/frame)

## Potential Bugs

1. RuntimeError: cannot import name '_compare_version' from 'torchmetrics.utilities.imports'. [Solution](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/11648)
2. OpenEXR-related errors loading `.exr` files (e.g. `cv2.imread` returning `None`), or numpy ABI mismatch crashes from `opencv-python`: the default `opencv-python` wheel does not always ship with OpenEXR support, and recent numpy 2.x builds are ABI-incompatible with some prebuilt opencv wheels. Use `opencv-python-headless==4.8.1.78` with `numpy<2` (both pinned in `requirements.txt`); verify with `cv2.getBuildInformation()` that it reports `OpenEXR: build`.
3. Computing LPIPS downloads pretrained VGG16 weights on first run (`torchvision`'s `vgg16` weights, cached under `~/.cache/torch/hub/checkpoints/`); make sure the machine has network access (possibly through a proxy) the first time you train/evaluate.

## Acknowledgements
part of our code is inherited from [InvRender](https://github.com/zju3dv/InvRender). We are grateful to the authors for releasing their code.

## Citation
```
@inproceedings{cheng2023structure,
  title={Structure from Duplicates: Neural Inverse Graphics from a Pile of Objects},
  author={Cheng, Tianhang and Ma, Wei-Chiu and Guan, Kaiyu and Torralba, Antonio and Wang, Shenlong},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```