#!/usr/bin/env python3
"""Render results/index.html from results.json (and meshes.json, if build_meshes.py has run)."""
import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent          # this checkout of the results branch
results = json.loads((OUT_DIR / "results.json").read_text())
mesh_index = json.loads((OUT_DIR / "meshes.json").read_text()) \
    if (OUT_DIR / "meshes.json").exists() else []
# Only offer a mesh the page can actually serve.
meshes = [m for m in mesh_index if (OUT_DIR / m["glb"]).exists()]
mesh_names = {m["name"] for m in meshes}
metrics_3d_by_name = {r["name"]: r.get("metrics_3d", {}) for r in results}
for m in meshes:
    m["metrics_3d"] = metrics_3d_by_name.get(m["name"], {}).get("world")


def fmt(v, nd=3, suffix=""):
    if v is None:
        return "—"
    if abs(v) < 1e-3 and v != 0:
        return f"{v:.3e}{suffix}"
    return f"{v:.{nd}f}{suffix}"


def mean_of(key, sub=None):
    vals = []
    for r in results:
        v = r["metrics"].get(key)
        if v is not None:
            vals.append(v)
    return sum(vals) / len(vals) if vals else None


METRIC_COLS = [
    ("rgb_psnr", "RGB PSNR↑", "{:.2f}"),
    ("rgb_ssim", "RGB SSIM↑", "{:.3f}"),
    ("rgb_lpips", "RGB LPIPS↓", "{:.3f}"),
    ("albedo_align_psnr", "Albedo PSNR↑ (aligned)", "{:.2f}"),
    ("albedo_align_ssim", "Albedo SSIM↑ (aligned)", "{:.3f}"),
    ("albedo_align_lpips", "Albedo LPIPS↓ (aligned)", "{:.3f}"),
    ("normal_abs_deg", "Normal Err↓ (°)", "{:.2f}"),
    ("roughness_mse", "Roughness MSE↓", "{:.2e}"),
    ("env_map_mse", "Env-map MSE↓", "{:.3f}"),
]

rows_html = []
for r in results:
    m = r["metrics"]
    tr = [f'<tr><td class="obj-name"><a href="#obj-{r["name"]}">{r["name"]}</a></td>']
    for key, _, fmt_s in METRIC_COLS:
        v = m.get(key)
        tr.append(f"<td>{fmt_s.format(v) if v is not None else '—'}</td>")
    tr.append(f"<td>{fmt(r['train'].get('total_hours'), 2, ' h')}</td>")
    tr.append("</tr>")
    rows_html.append("".join(tr))

mean_row = ['<tr class="mean-row"><td class="obj-name">mean (15 objects)</td>']
for key, _, fmt_s in METRIC_COLS:
    v = mean_of(key)
    mean_row.append(f"<td>{fmt_s.format(v) if v is not None else '—'}</td>")
total_h = [r["train"].get("total_hours") for r in results if r["train"].get("total_hours") is not None]
mean_row.append(f"<td>{fmt(sum(total_h) / len(total_h), 2, ' h') if total_h else '—'}</td>")
mean_row.append("</tr>")

header_cells = "".join(f"<th>{label}</th>" for _, label, _ in METRIC_COLS)

# --- 3D: mesh vs. Blender ground truth --------------------------------------------------------
# Distances are relative to the ground-truth bounding box diagonal (objects differ in metric size by
# more than 10x). 'world' compares the whole pile in the Blender world frame, 'local' compares one
# instance in the object's own frame; the world frame is the honest one, the local frame divides by a
# single object's diagonal and hides part of the pose error.
METRIC_3D_COLS = [
    ("chamfer_l1_relative", "CD-L1↓ (% diag)", lambda v: f"{100 * v:.3f}"),
    ("accuracy_relative", "Accuracy↓ (% diag)", lambda v: f"{100 * v:.3f}"),
    ("completeness_relative", "Completeness↓ (% diag)", lambda v: f"{100 * v:.3f}"),
    ("f_score@0.005", "F@0.5%↑", lambda v: f"{v:.3f}"),
    ("f_score@0.01", "F@1%↑", lambda v: f"{v:.3f}"),
    ("f_score@0.02", "F@2%↑", lambda v: f"{v:.3f}"),
    ("normal_consistency", "Normal cons.↑", lambda v: f"{v:.3f}"),
    ("pose_corner_spread_relative", "Pose spread (% diag)", lambda v: f"{100 * v:.3f}"),
]


def metrics_3d_of(entry, frame):
    """The 3D metrics of one object in one frame, with the pose spread made relative."""
    m = dict(entry.get("metrics_3d", {}).get(frame) or {})
    if m.get("diagonal"):
        m["pose_corner_spread_relative"] = m["pose_corner_spread"] / m["diagonal"]
    return m


def rows_3d(frame):
    """Table rows of the 3D metrics, plus a mean row, for the objects that have ground truth."""
    have = [r for r in results if metrics_3d_of(r, frame)]
    rows = []
    for r in have:
        m = metrics_3d_of(r, frame)
        cells = [f'<tr><td class="obj-name"><a href="#obj-{r["name"]}">{r["name"]}</a></td>']
        for key, _, fmt_v in METRIC_3D_COLS:
            v = m.get(key)
            cells.append(f"<td>{fmt_v(v) if v is not None else '—'}</td>")
        cells.append("</tr>")
        rows.append("".join(cells))
    mean = [f'<tr class="mean-row"><td class="obj-name">mean ({len(have)} objects)</td>']
    for key, _, fmt_v in METRIC_3D_COLS:
        vals = [metrics_3d_of(r, frame).get(key) for r in have]
        vals = [v for v in vals if v is not None]
        mean.append(f"<td>{fmt_v(sum(vals) / len(vals)) if vals else '—'}</td>")
    mean.append("</tr>")
    return "".join(rows) + "".join(mean), len(have)


header_cells_3d = "".join(f"<th>{label}</th>" for _, label, _ in METRIC_3D_COLS)
rows_3d_world, n_3d = rows_3d("world")
rows_3d_local, _ = rows_3d("local")
no_3d = sorted(r["name"] for r in results if not metrics_3d_of(r, "world"))

# --- the mesh viewer --------------------------------------------------------------------------
mesh_chips = "".join(
    f'<a href="#mesh-viewer" data-object="{m["name"]}">{m["name"]}</a>' for m in meshes)
mesh_data = json.dumps(meshes, indent=1)

viewer_section = f"""
<h2>Reconstructed geometry — drag to rotate</h2>
<div class="panel">
  <p class="note">
    Marching cubes at resolution 512 on the canonical SDF of the <code>Mat</code> checkpoint
    (<code>exp_runner.py --to_mesh</code>), decimated to 30k faces for the web
    (<code>build_meshes.py</code>); the colours are the predicted diffuse albedo, per vertex.
    Drag to rotate, scroll to zoom, right-drag to pan.
    The {n_3d} synthetic objects are rotated upright using the Blender ground truth; the
    {len(no_3d)} real-world captures have nothing that defines "up", so they are shown in the
    network's canonical frame and come out tilted.
  </p>
  <div class="toc viewer-picker" id="mesh-picker">{mesh_chips}</div>
  <div class="viewer-shell">
    <div id="mesh-viewer"></div>
    <div id="viewer-status"></div>
  </div>
  <div class="viewer-bar">
    <div id="viewer-info"></div>
    <div class="viewer-controls">
      <label><input type="checkbox" id="viewer-spin" checked> auto-spin</label>
      <label><input type="checkbox" id="viewer-albedo" checked> albedo colours</label>
      <button id="viewer-reset">reset view</button>
    </div>
  </div>
</div>
<script type="application/json" id="mesh-data">{mesh_data}</script>
<script type="importmap">
{{"imports": {{"three": "./vendor/three/three.module.min.js"}}}}
</script>
<script type="module" src="viewer.js"></script>
""" if meshes else ""

IMG_ROWS = [
    ("rgb", "RGB"),
    ("diffuse", "Albedo (diffuse)"),
    ("normal", "Normal"),
    ("rough", "Roughness"),
    ("metal", "Metallic"),
]

gallery_sections = []
for r in results:
    imgs = r["images"]
    m = r["metrics"]
    cells = []
    for key, label in IMG_ROWS:
        gt = imgs.get(f"{key}_gt")
        ours = imgs.get(f"{key}_our_0")
        if not gt or not ours:
            continue
        cells.append(f"""
        <div class="img-pair">
          <div class="img-pair-label">{label}</div>
          <div class="img-pair-imgs">
            <figure><img src="{gt}" loading="lazy"><figcaption>GT</figcaption></figure>
            <figure><img src="{ours}" loading="lazy"><figcaption>Ours</figcaption></figure>
          </div>
        </div>""")

    stats_line = (
        f"RGB PSNR {fmt(m.get('rgb_psnr'), 2)} · SSIM {fmt(m.get('rgb_ssim'))} · LPIPS {fmt(m.get('rgb_lpips'))} &nbsp;|&nbsp; "
        f"Albedo PSNR {fmt(m.get('albedo_align_psnr'), 2)} (aligned) &nbsp;|&nbsp; "
        f"Normal err {fmt(m.get('normal_abs_deg'), 2)}° &nbsp;|&nbsp; "
        f"Roughness MSE {fmt(m.get('roughness_mse'))} &nbsp;|&nbsp; "
        f"Train {fmt(r['train'].get('total_hours'), 2)} h"
    )
    m3 = metrics_3d_of(r, "world")
    if m3:
        stats_line += (f" &nbsp;|&nbsp; Mesh CD-L1 {100 * m3['chamfer_l1_relative']:.3f}% of diag"
                       f" · F@1% {m3['f_score@0.01']:.3f}")
    mesh_link = (f'<a class="view-mesh" href="#mesh-viewer" data-object="{r["name"]}">'
                 f'view the 3D mesh ▸</a>') if r["name"] in mesh_names else ""

    gallery_sections.append(f"""
    <section class="obj-section" id="obj-{r['name']}">
      <h3>{r['name']} {mesh_link}</h3>
      <div class="stats-line">{stats_line}</div>
      <div class="img-row">{''.join(cells)}</div>
    </section>""")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SfD Training &amp; Eval Report — DuplicateSingleImage (15 objects)</title>
<style>
  :root {{
    --bg: #0f1117; --panel: #171a23; --border: #2a2e3a; --text: #e5e7eb;
    --muted: #9aa1b2; --accent: #6ea8fe;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    margin: 0; padding: 0 0 60px;
  }}
  header {{
    padding: 32px 40px 20px; border-bottom: 1px solid var(--border);
    background: linear-gradient(180deg, #1a1d29, var(--bg));
  }}
  header h1 {{ margin: 0 0 6px; font-size: 26px; }}
  header p {{ margin: 4px 0; color: var(--muted); font-size: 14px; }}
  main {{ padding: 30px 40px; max-width: 1400px; margin: 0 auto; }}
  h2 {{ font-size: 19px; border-left: 4px solid var(--accent); padding-left: 10px; margin-top: 46px; }}
  .panel {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 20px; overflow-x: auto;
  }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13.5px; white-space: nowrap; }}
  th, td {{ padding: 8px 12px; text-align: right; border-bottom: 1px solid var(--border); }}
  th:first-child, td:first-child {{ text-align: left; position: sticky; left: 0; background: var(--panel); }}
  thead th {{ color: var(--muted); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: .02em; }}
  tbody tr:hover {{ background: #1f2330; }}
  .obj-name a {{ color: var(--accent); text-decoration: none; }}
  .mean-row td {{ font-weight: 700; border-top: 2px solid var(--accent); color: #fff; }}
  .plots {{ display: flex; gap: 20px; flex-wrap: wrap; margin-top: 16px; }}
  .plots img {{ max-width: 100%; border-radius: 8px; border: 1px solid var(--border); }}
  .plots .plot-card {{ flex: 1 1 560px; }}
  .obj-section {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 18px 20px; margin-top: 16px;
  }}
  .obj-section h3 {{ margin: 0 0 6px; font-size: 17px; text-transform: capitalize; }}
  .stats-line {{ color: var(--muted); font-size: 12.5px; margin-bottom: 14px; }}
  .img-row {{ display: flex; gap: 18px; flex-wrap: wrap; }}
  .img-pair-label {{ font-size: 12px; color: var(--muted); margin-bottom: 6px; text-transform: uppercase; letter-spacing: .03em; }}
  .img-pair-imgs {{ display: flex; gap: 6px; }}
  .img-pair-imgs figure {{ margin: 0; text-align: center; }}
  .img-pair-imgs img {{ width: 140px; height: 140px; object-fit: cover; border-radius: 6px; border: 1px solid var(--border); background: #000; }}
  .img-pair-imgs figcaption {{ font-size: 10px; color: var(--muted); margin-top: 3px; }}
  .toc {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }}
  .toc a {{
    color: var(--accent); text-decoration: none; font-size: 12.5px; padding: 4px 10px;
    border: 1px solid var(--border); border-radius: 999px;
  }}
  .toc a:hover {{ background: #1f2330; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; background: #1e3a2e; color: #7ee2a8; font-size: 11px; margin-left: 8px; }}
  .note {{ color: var(--muted); font-size: 13px; line-height: 1.55; margin: 0 0 14px; max-width: 100ch; }}
  .note code {{ color: #cbd3e6; }}
  .viewer-shell {{ position: relative; }}
  #mesh-viewer {{
    width: 100%; height: min(62vh, 560px); min-height: 320px; border-radius: 8px;
    border: 1px solid var(--border); background: #0d0f16; overflow: hidden; cursor: grab;
  }}
  #mesh-viewer:active {{ cursor: grabbing; }}
  #mesh-viewer canvas {{ display: block; width: 100%; height: 100%; }}
  #viewer-status {{
    position: absolute; top: 12px; left: 14px; color: var(--muted); font-size: 12.5px;
    pointer-events: none;
  }}
  .viewer-picker {{ margin-bottom: 12px; }}
  .viewer-picker a.active {{ background: #223; border-color: var(--accent); color: #fff; }}
  .viewer-bar {{
    display: flex; gap: 16px; align-items: center; justify-content: space-between;
    flex-wrap: wrap; margin-top: 12px; color: var(--muted); font-size: 12.5px;
  }}
  .viewer-controls {{ display: flex; gap: 14px; align-items: center; white-space: nowrap; }}
  .viewer-controls label {{ display: flex; gap: 5px; align-items: center; cursor: pointer; }}
  .viewer-controls button {{
    background: #1f2330; color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: 4px 10px; font-size: 12.5px; cursor: pointer;
  }}
  .viewer-controls button:hover {{ border-color: var(--accent); }}
  .view-mesh {{ color: var(--accent); text-decoration: none; font-size: 12px; font-weight: 400; margin-left: 8px; text-transform: none; }}
  details.local-frame {{ margin-top: 14px; }}
  details.local-frame summary {{ color: var(--accent); font-size: 12.5px; cursor: pointer; }}
  details.local-frame > div {{ margin-top: 12px; overflow-x: auto; }}
  footer {{ text-align: center; color: var(--muted); font-size: 12px; margin-top: 50px; }}
</style>
</head>
<body>
<header>
  <h1>SfD Training &amp; Evaluation Report <span class="badge">15/15 objects · Geo→Vis→Mat complete</span></h1>
  <p>Dataset: DuplicateSingleImage · Pipeline: <code>cmd_train.sh</code> (Geo/Vis/Mat, 4-GPU shard) → <code>cmd_eval.sh</code> (held-out test view, mesh export, 3D metrics)</p>
  <p>Image metrics computed on the object mask of the single held-out test frame per object. Albedo metrics reported with per-channel scale alignment (standard for relit-albedo evaluation since absolute scale is unobservable). Mesh metrics computed against the Blender ground truth of the {n_3d} synthetic objects.</p>
</header>
<main>

<h2>Summary — image metrics per object</h2>
<div class="panel">
<table>
<thead><tr><th>Object</th>{header_cells}<th>Train time</th></tr></thead>
<tbody>
{''.join(rows_html)}
{''.join(mean_row)}
</tbody>
</table>
</div>

<h2>Visualizations</h2>
<div class="plots">
  <div class="plot-card panel"><img src="metrics_plot.png"></div>
  <div class="plot-card panel"><img src="training_time_plot.png"></div>
</div>

<h2>3D metrics — mesh vs. Blender ground truth</h2>
<div class="panel">
  <p class="note">
    Marching cubes on the canonical SDF of the <code>Mat</code> checkpoint, replicated to every
    instance and compared against the mesh exported straight out of the <code>.blend</code>
    (<code>scripts/eval_mesh_3d.py</code>, 200k surface samples per mesh, no ICP — the alignment is
    the analytic SfM↔Blender similarity of <code>utils/blender_align.py</code>).
    Distances are given as a percentage of the ground-truth bounding box diagonal, since the objects
    differ in metric size by more than 10×.
    <b>Pose spread</b> is how far the per-instance canonical→Blender transforms disagree with each
    other, in the same units: it is an upper bound on how much of the distance is SfM pose error
    rather than shape error (it is measured at the corners of the canonical unit cube, which is
    wider than the objects). At {n_3d} objects with a mean CD-L1 of the same order as that spread,
    these distances say about as much about the poses as about the geometry.
    The {len(no_3d)} real-world captures ({', '.join(no_3d)}) ship no <code>.blend</code>, so they
    have no 3D metrics at all.
  </p>
<table>
<thead><tr><th>Object</th>{header_cells_3d}</tr></thead>
<tbody>
{rows_3d_world}
</tbody>
</table>
<details class="local-frame">
  <summary>Single instance, in the object's own frame (local frame)</summary>
  <div>
    <p class="note">
      The same meshes compared one instance at a time, in the frame of the Blender object. The
      denominator is now a single object's diagonal instead of the whole pile's, ~4× smaller, so the
      percentages are ~4× larger; the pose spread grows with it. The world-frame table above is the
      one to quote.
    </p>
    <table>
    <thead><tr><th>Object</th>{header_cells_3d}</tr></thead>
    <tbody>
    {rows_3d_local}
    </tbody>
    </table>
  </div>
</details>
</div>
<div class="plots">
  <div class="plot-card panel" style="flex-basis: 100%"><img src="metrics_3d_plot.png"></div>
</div>
{viewer_section}
<h2>Per-object renders (GT vs. Ours)</h2>
<div class="toc">
  {''.join(f'<a href="#obj-{r["name"]}">{r["name"]}</a>' for r in results)}
</div>
{''.join(gallery_sections)}

<footer>Generated from cmd_train.sh / cmd_eval.sh output in /mnt/task_runtime · <code>results.json</code> has the raw image metrics and the full 3D metrics, <code>meshes.json</code> the mesh inventory · the checkpoints, evaluation output and meshes are released at <a href="https://huggingface.co/TianhangCheng7/DuplicateWeight">TianhangCheng7/DuplicateWeight</a> · 3D view powered by a vendored <a href="vendor/three/LICENSE">three.js r170</a></footer>
</main>
</body>
</html>
"""

(OUT_DIR / "index.html").write_text(html)
print("wrote", OUT_DIR / "index.html")
