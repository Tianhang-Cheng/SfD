# Omnidata (trimmed vendored copy)

This folder is **not** the full [Omnidata](https://github.com/EPFL-VILAB/omnidata) repository. It only
keeps the modules that `preprocess/8_extract_monocular_cues.py` imports in order to run the
pretrained surface-normal / depth network:

```
omnidata_tools/torch/modules/   # DPT + MiDaS backbones (modules.midas.dpt_depth.DPTDepthModel)
omnidata_tools/torch/data/      # data.transforms.get_transform and its task configs
omnidata_tools/torch/tools/     # upstream checkpoint download scripts
```

Everything else from upstream (the paper code, the Blender annotator, the dataset tooling, the docs
and the demo GIFs — about 330 MB) was removed to keep this repository small. If you need any of it,
clone the upstream repo instead.

## Pretrained checkpoint

The monocular-cue network weights are not redistributed here. Download
`omnidata_dpt_normal_v2.ckpt` from [Omnidata](https://github.com/EPFL-VILAB/omnidata) (or run
`omnidata_tools/torch/tools/download_surface_normal_models.sh`) and put it in
`omnidata_tools/torch/pretrained_models/`. Step 8 of the preprocessing pipeline is skipped when the
checkpoint is missing.

## Citing

```
@inproceedings{eftekhar2021omnidata,
  title={Omnidata: A Scalable Pipeline for Making Multi-Task Mid-Level Vision Datasets from 3D Scans},
  author={Eftekhar, Ainaz and Sax, Alexander and Malik, Jitendra and Zamir, Amir},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10786--10796},
  year={2021}
}
```

See `LICENSE` for the upstream terms.
