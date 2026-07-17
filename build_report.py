#!/usr/bin/env python3
"""Collect SfD train/eval results into results.json, copy gallery images, and plot metrics."""
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/mnt/task_runtime")
SFD_EXPS = ROOT / "SfD" / "exps"
LOG_DIR = ROOT / "train_logs"
OUT_DIR = ROOT / "results"
ASSETS_DIR = OUT_DIR / "assets"

SAMPLES = ["airplane", "box", "cake", "cash", "cheese", "cleaner", "clock",
           "coffee", "cola", "fire", "gitar", "potato", "sign", "tin", "yogurt"]

NUM_RE = re.compile(r"=\s*([-\d.eE+]+)")
TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")


def read_num(path: Path):
    if not path.exists():
        return None
    txt = path.read_text()
    m = NUM_RE.search(txt)
    return float(m.group(1)) if m else None


def latest_eval_run(sample: str) -> Path | None:
    d = SFD_EXPS / f"Mat-{sample}-eval"
    if not d.exists():
        return None
    runs = sorted(p for p in d.iterdir() if p.is_dir())
    return runs[-1] if runs else None


def parse_train_log(sample: str):
    log = LOG_DIR / f"{sample}.log"
    stages = {}
    if not log.exists():
        return stages
    with log.open() as f:
        for line in f:
            m = TS_RE.match(line)
            if not m:
                continue
            ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            if "stage 1/3 Geo" in line:
                stages["geo_start"] = ts
            elif "stage 2/3 Vis" in line:
                stages["vis_start"] = ts
            elif "stage 3/3 Mat" in line:
                stages["mat_start"] = ts
            elif "DONE (all 3 stages)" in line:
                stages["done"] = ts
            elif "FAILED" in line:
                stages["failed_at"] = ts
    return stages


def hours(a, b):
    return round((b - a).total_seconds() / 3600, 2) if a and b else None


def collect():
    results = []
    for sample in SAMPLES:
        run_dir = latest_eval_run(sample)
        entry = {"name": sample, "eval_run": run_dir.name if run_dir else None}

        if run_dir:
            ev = run_dir / "evals_value"
            entry["metrics"] = {
                "rgb_psnr": read_num(ev / "rgb_psnr_obj_mask.txt"),
                "rgb_ssim": read_num(ev / "rgb_ssim_obj_mask.txt"),
                "rgb_lpips": read_num(ev / "rgb_lpips_obj_mask.txt"),
                "albedo_psnr": read_num(ev / "albedo_psnr_obj_mask.txt"),
                "albedo_ssim": read_num(ev / "albedo_ssim_obj_mask.txt"),
                "albedo_lpips": read_num(ev / "albedo_lpips_obj_mask.txt"),
                "albedo_align_psnr": read_num(ev / "albedo_align_psnr_obj_mask.txt"),
                "albedo_align_ssim": read_num(ev / "albedo_align_ssim_obj_mask.txt"),
                "albedo_align_lpips": read_num(ev / "albedo_align_lpips_obj_mask.txt"),
                "normal_abs_deg": read_num(ev / "normal_abs_obj_mask.txt"),
                "roughness_mse": read_num(ev / "roughness_mse_obj_mask.txt"),
                "env_map_mse": read_num(run_dir / "env_map_mse.txt"),
                "eval_run_time_s": read_num(run_dir / "run_time.txt"),
            }
            # copy gallery images
            src_img = run_dir / "evals_image"
            dst_img = ASSETS_DIR / sample
            dst_img.mkdir(parents=True, exist_ok=True)
            imgs = {}
            if src_img.exists():
                for p in sorted(src_img.glob("*.png")):
                    shutil.copy(p, dst_img / p.name)
                    imgs[p.stem] = f"assets/{sample}/{p.name}"
            entry["images"] = imgs
        else:
            entry["metrics"] = {}
            entry["images"] = {}

        stages = parse_train_log(sample)
        entry["train"] = {
            "geo_hours": hours(stages.get("geo_start"), stages.get("vis_start")),
            "vis_hours": hours(stages.get("vis_start"), stages.get("mat_start")),
            "mat_hours": hours(stages.get("mat_start"), stages.get("done")),
            "total_hours": hours(stages.get("geo_start"), stages.get("done")),
            "status": "done" if "done" in stages else ("failed" if "failed_at" in stages else "unknown"),
        }
        results.append(entry)

    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    return results


def plot(results):
    names = [r["name"] for r in results]
    x = np.arange(len(names))

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle("SfD Final Performance per Object (test view, object mask)", fontsize=15)

    def bar(ax, key, title, ylabel, fmt="{:.2f}", color="tab:blue"):
        vals = [r["metrics"].get(key) for r in results]
        vals_clean = [v if v is not None else 0 for v in vals]
        ax.bar(x, vals_clean, color=color)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
        mean_v = np.mean([v for v in vals if v is not None])
        ax.axhline(mean_v, color="red", linestyle="--", linewidth=1)
        ax.text(len(names) - 0.5, mean_v, f" mean={fmt.format(mean_v)}", color="red", va="bottom", fontsize=8)

    bar(axes[0, 0], "rgb_psnr", "RGB PSNR (higher better)", "dB", color="tab:blue")
    bar(axes[0, 1], "rgb_ssim", "RGB SSIM (higher better)", "SSIM", color="tab:green")
    bar(axes[0, 2], "rgb_lpips", "RGB LPIPS (lower better)", "LPIPS", color="tab:red")
    bar(axes[1, 0], "albedo_align_psnr", "Albedo PSNR, scale-aligned (higher better)", "dB", color="tab:orange")
    bar(axes[1, 1], "normal_abs_deg", "Normal Angular Error (lower better)", "degrees", color="tab:purple")
    bar(axes[1, 2], "roughness_mse", "Roughness MSE (lower better)", "MSE", fmt="{:.2e}", color="tab:brown")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_DIR / "metrics_plot.png", dpi=130)
    plt.close(fig)

    # training time breakdown stacked bar
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    geo = [r["train"].get("geo_hours") or 0 for r in results]
    vis = [r["train"].get("vis_hours") or 0 for r in results]
    mat = [r["train"].get("mat_hours") or 0 for r in results]
    ax2.bar(x, geo, label="Geo", color="#4c72b0")
    ax2.bar(x, vis, bottom=geo, label="Vis", color="#dd8452")
    ax2.bar(x, mat, bottom=np.array(geo) + np.array(vis), label="Mat", color="#55a868")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax2.set_ylabel("hours")
    ax2.set_title("Training time per stage (Geo / Vis / Mat)")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "training_time_plot.png", dpi=130)
    plt.close(fig2)


if __name__ == "__main__":
    results = collect()
    plot(results)
    print(f"Collected {len(results)} samples")
    for r in results:
        print(r["name"], r["metrics"].get("rgb_psnr"), r["train"]["total_hours"])
