"""
Compare a render with the training image of a processed object.

Works on any pair of images, so it is also useful for comparing the network's own renders:

    python scripts/compare_render.py --render /tmp/coffee_gt.exr \\
        --data_split_dir hf_data/train_split/coffee --output /tmp/coffee_compare.png

Both images are brought into the same linear space, optionally exposure matched, and then
compared with PSNR/SSIM over the whole frame and inside the object mask. The reported
sub-pixel shift is the interesting number when checking a *camera* convention: a systematic
offset means the pose or the intrinsics are off, while a shift of ~0 with a brightness
mismatch means the geometry lines up and only the shading differs.
"""

import argparse
import os
import sys
from typing import Dict, Optional, Tuple

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import imageio.v2 as imageio
import numpy as np

from utils import rend_util


def parse_args() -> argparse.Namespace:
    """
    Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--render', type=str, required=True,
                        help='image to check, .exr (linear) or .png (sRGB)')
    parser.add_argument('--reference', type=str, default='',
                        help='image to compare against; taken from --data_split_dir by default')
    parser.add_argument('--data_split_dir', type=str, default='',
                        help='processed object directory, used for the reference image and the '
                             'object mask')
    parser.add_argument('--mask', type=str, default='',
                        help='optional mask image; train/000_instance_seg.png by default')
    parser.add_argument('--output', type=str, default='',
                        help='where to write the side by side figure (png)')
    parser.add_argument('--exposure_match', default=True, action=argparse.BooleanOptionalAction,
                        help='scale the render so that its mean matches the reference inside '
                             'the mask, which takes an unknown HDRI strength out of the picture')
    parser.add_argument('--gamma', type=float, default=2.2,
                        help='gamma used to display and to compute the image metrics')
    return parser.parse_args()


def load_linear_image(path: str, gamma: float = 2.2) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Read an image as linear RGB, plus its alpha channel if it has one.

    Args:
        path: ``.exr`` (already linear) or 8/16 bit file (assumed sRGB-ish, gamma decoded).
        gamma: gamma to undo for non EXR inputs.

    Returns:
        image: ``[h,w,3]`` float32 linear RGB.
        alpha: ``[h,w]`` float32 alpha, or None.

    Raises:
        SystemExit: if the file is missing or cannot be decoded; OpenCV only warns and hands
            back an empty array in that case, which would fail much later and confusingly.
    """
    if not os.path.exists(path):
        raise SystemExit('no such image: {}\n'
                         '(if the render step printed "blender: command not found", there is '
                         'nothing to compare yet -- install Blender first)'.format(path))
    if path.lower().endswith('.exr'):
        image = np.asarray(rend_util.load_exr(path), dtype=np.float32)
        # imageio tone maps an exr down to 8 bit rgb, so the alpha has to come from cv2
        import cv2
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        alpha = np.asarray(raw, dtype=np.float32)[..., 3] \
            if raw is not None and raw.ndim == 3 and raw.shape[2] == 4 else None
    else:
        raw = imageio.imread(path)
        maximum = 65535.0 if raw.dtype == np.uint16 else 255.0
        raw = np.asarray(raw, dtype=np.float32) / maximum
        alpha = raw[..., 3] if raw.ndim == 3 and raw.shape[2] == 4 else None
        image = raw[..., :3] ** gamma
    if image.ndim != 3 or image.shape[2] < 3:
        raise SystemExit('could not decode {} as an rgb image (got shape {})'
                         .format(path, image.shape))
    return np.nan_to_num(image[..., :3]), alpha


def load_object_mask(data_split_dir: str, mask_path: str = '') -> Optional[np.ndarray]:
    """
    Read the union mask of the object instances.

    Args:
        data_split_dir: processed object directory, or ``''``.
        mask_path: explicit mask image; overrides ``data_split_dir``.

    Returns:
        ``[h,w]`` boolean mask, or None if neither source is available.
    """
    if not mask_path and data_split_dir:
        for name in ['000_instance_seg.png', '000_mask.png']:
            candidate = os.path.join(data_split_dir, 'train', name)
            if os.path.exists(candidate):
                mask_path = candidate
                break
    if not mask_path:
        return None
    mask = np.asarray(imageio.imread(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask > 0


def tonemap(image: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """
    Display transform: clip to ``[0,1]`` after applying the inverse gamma.

    Args:
        image: linear RGB array.
        gamma: display gamma.

    Returns:
        An array of the same shape in ``[0,1]``.
    """
    return np.clip(np.clip(image, 0.0, None) ** (1.0 / gamma), 0.0, 1.0)


def image_metrics(render: np.ndarray, reference: np.ndarray,
                  mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    PSNR/SSIM/MAE of two display ready images, optionally restricted to a mask.

    Args:
        render: ``[h,w,3]`` array in ``[0,1]``.
        reference: ``[h,w,3]`` array in ``[0,1]``.
        mask: ``[h,w]`` boolean mask, or None for the whole image.

    Returns:
        A dict with ``psnr``, ``mae`` and, only when no mask is given, ``ssim`` (SSIM is a
        windowed statistic, so it has no meaningful masked version).
    """
    if mask is None:
        difference = render - reference
    else:
        difference = (render - reference)[mask]
    mse = float(np.mean(difference ** 2))
    metrics = {'psnr': float('inf') if mse == 0 else float(-10.0 * np.log10(mse)),
               'mae': float(np.mean(np.abs(difference)))}
    if mask is None:
        from skimage.metrics import structural_similarity
        metrics['ssim'] = float(structural_similarity(reference, render, channel_axis=2,
                                                      data_range=1.0))
    return metrics


def estimate_shift(render: np.ndarray, reference: np.ndarray) -> Tuple[float, float]:
    """
    Sub-pixel translation between two images, by phase correlation of their luminance.

    Args:
        render: ``[h,w,3]`` array in ``[0,1]``.
        reference: ``[h,w,3]`` array in ``[0,1]``.

    Returns:
        The ``(row, column)`` shift that would move the render onto the reference.
    """
    from skimage.registration import phase_cross_correlation

    shift, _, _ = phase_cross_correlation(reference.mean(axis=2), render.mean(axis=2),
                                          upsample_factor=20)
    return float(shift[0]), float(shift[1])


def save_comparison(path: str, render: np.ndarray, reference: np.ndarray,
                    metrics: Dict[str, float]) -> str:
    """
    Write a render / reference / absolute difference figure.

    Args:
        path: destination png.
        render: ``[h,w,3]`` display ready render.
        reference: ``[h,w,3]`` display ready reference.
        metrics: metrics to put in the title.

    Returns:
        The path that was written.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(render)
    axes[0].set_title('render')
    axes[1].imshow(reference)
    axes[1].set_title('training image')
    difference = np.abs(render - reference).mean(axis=2)
    handle = axes[2].imshow(difference, cmap='inferno', vmin=0.0, vmax=max(0.05,
                                                                          difference.max()))
    axes[2].set_title('|difference|, psnr {:.2f} dB, ssim {:.3f}'.format(
        metrics['psnr'], metrics['ssim']))
    figure.colorbar(handle, ax=axes[2], fraction=0.046)
    for axis in axes:
        axis.axis('off')
    figure.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    figure.savefig(path, dpi=120, bbox_inches='tight', pad_inches=0.05)
    plt.close(figure)
    return path


def main() -> None:
    """Compare the two images and print the metrics."""
    args = parse_args()

    reference_path = args.reference
    if not reference_path:
        if not args.data_split_dir:
            raise SystemExit('pass either --reference or --data_split_dir')
        for name in ['000_rgb.exr', '000_rgb.png']:
            candidate = os.path.join(args.data_split_dir, 'train', name)
            if os.path.exists(candidate):
                reference_path = candidate
                break
        if not reference_path:
            raise SystemExit('no train/000_rgb.exr or .png in {}'.format(args.data_split_dir))

    render, render_alpha = load_linear_image(args.render, args.gamma)
    reference, _ = load_linear_image(reference_path, args.gamma)
    print('render    {} {}'.format(args.render, render.shape))
    print('reference {} {}'.format(reference_path, reference.shape))
    if render.shape != reference.shape:
        raise SystemExit('the two images have different shapes; render at the resolution of '
                         'the training image ({}x{})'.format(reference.shape[1],
                                                             reference.shape[0]))

    mask = load_object_mask(args.data_split_dir, args.mask)
    if mask is not None and mask.shape != reference.shape[:2]:
        mask = None
    scale = 1.0
    if args.exposure_match:
        region = mask if mask is not None else np.ones(reference.shape[:2], dtype=bool)
        numerator = float(reference[region].mean())
        denominator = float(render[region].mean())
        scale = 1.0 if denominator <= 1e-8 else numerator / denominator
        print('exposure match: scaled the render by {:.4f}'.format(scale))
    render = render * scale

    render_display = tonemap(render, args.gamma)
    reference_display = tonemap(reference, args.gamma)

    whole = image_metrics(render_display, reference_display)
    print('whole image : psnr {psnr:.2f} dB, ssim {ssim:.4f}, mae {mae:.4f}'.format(**whole))
    if mask is not None:
        inside = image_metrics(render_display, reference_display, mask)
        print('object mask : psnr {psnr:.2f} dB, mae {mae:.4f}'.format(**inside))
        if render_alpha is not None:
            predicted = render_alpha > 0.5
            union = np.logical_or(predicted, mask).sum()
            iou = 0.0 if union == 0 else float(np.logical_and(predicted, mask).sum() / union)
            print('silhouette  : IoU {:.4f} (render alpha vs the dataset mask)'.format(iou))

    row, column = estimate_shift(render_display, reference_display)
    print('shift       : {:+.2f} px rows, {:+.2f} px columns (0 means the cameras agree)'
          .format(row, column))

    output = args.output or os.path.splitext(args.render)[0] + '_vs_train.png'
    print('wrote {}'.format(save_comparison(output, render_display, reference_display, whole)))


if __name__ == '__main__':
    main()
