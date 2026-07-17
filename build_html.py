#!/usr/bin/env python3
"""Render results/index.html from results.json."""
import json
from pathlib import Path

OUT_DIR = Path("/mnt/task_runtime/results")
results = json.loads((OUT_DIR / "results.json").read_text())


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

    gallery_sections.append(f"""
    <section class="obj-section" id="obj-{r['name']}">
      <h3>{r['name']}</h3>
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
  footer {{ text-align: center; color: var(--muted); font-size: 12px; margin-top: 50px; }}
</style>
</head>
<body>
<header>
  <h1>SfD Training &amp; Evaluation Report <span class="badge">15/15 objects · Geo→Vis→Mat complete</span></h1>
  <p>Dataset: DuplicateSingleImage · Pipeline: <code>cmd_train.sh</code> (Geo/Vis/Mat, 4-GPU shard) → <code>cmd_eval.sh</code> (held-out test view)</p>
  <p>Metrics computed on the object mask of the single held-out test frame per object. Albedo metrics reported with per-channel scale alignment (standard for relit-albedo evaluation since absolute scale is unobservable).</p>
</header>
<main>

<h2>Summary — final metrics per object</h2>
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

<h2>Per-object renders (GT vs. Ours)</h2>
<div class="toc">
  {''.join(f'<a href="#obj-{r["name"]}">{r["name"]}</a>' for r in results)}
</div>
{''.join(gallery_sections)}

<footer>Generated from cmd_train.sh / cmd_eval.sh output in /mnt/task_runtime · results.json has the raw numbers</footer>
</main>
</body>
</html>
"""

(OUT_DIR / "index.html").write_text(html)
print("wrote", OUT_DIR / "index.html")
