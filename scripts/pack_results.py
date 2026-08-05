"""
Pack the trained runs under ``exps/`` into one zip for the release on the Hub.

The full ``exps/`` tree of a 15 object sweep is ~8.6 GB, most of it training byproducts. This
script keeps what a user of the release needs -- the final weights, the evaluation output and the
exported meshes -- and drops what only mattered while training:

* ``checkpoints/ModelParameters/`` keeps ``latest.pth`` only; the periodic ``<iter>.pth`` snapshots
  are dropped (``--keep_all_checkpoints`` keeps them),
* the optimizer state (``NEUS*OptimizerParameters/``) is dropped, since only a training *resume*
  reads it and it is ~60 % of the checkpoint bytes (``--include_optimizer`` keeps it),
* ``plots/`` (the per-iteration training visualisations, 3.1 GB) and the tensorboard event files
  are dropped (``--include_plots`` / ``--include_tb`` keep them).

The archive mirrors the repository layout, so unzipping it in the repository root puts every run
back where ``exp_runner.py`` looks for it:

    python scripts/pack_results.py --output /tmp/SfD_results.zip
    unzip -d /path/to/SfD /tmp/SfD_results.zip   # -> exps/... and train_logs/...

A ``MANIFEST.md`` describing the contents is generated into the archive.
"""

import argparse
import fnmatch
import os
import re
import subprocess
import zipfile
from typing import Dict, List, Optional, Tuple

# Files that are already compressed; deflating them again costs time and saves nothing.
STORED_SUFFIXES = ('.pth', '.npz', '.exr', '.png', '.jpg', '.jpeg', '.zip', '.gz', '.glb')

CHECKPOINT_SNAPSHOT = re.compile(r'^\d+\.pth$')


def parse_args() -> argparse.Namespace:
    """
    Parse the command line.

    Returns:
        The parsed arguments.
    """
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--exps_dir', type=str, default=os.path.join(repo, 'exps'),
                        help='the exps/ directory to pack')
    parser.add_argument('--log_dir', type=str, default=os.path.join(repo, 'train_logs'),
                        help='per-object training/eval logs to include; "" skips them')
    parser.add_argument('--output', type=str, required=True, help='zip file to write')
    parser.add_argument('--keep_all_checkpoints', default=False, action='store_true',
                        help='keep the periodic <iter>.pth snapshots as well as latest.pth')
    parser.add_argument('--include_optimizer', default=False, action='store_true',
                        help='keep the optimizer state, i.e. make training resumable')
    parser.add_argument('--include_plots', default=False, action='store_true',
                        help='keep the plots/ training visualisations')
    parser.add_argument('--include_tb', default=False, action='store_true',
                        help='keep the tensorboard event files')
    parser.add_argument('--dry_run', default=False, action='store_true',
                        help='report what would be packed without writing the zip')
    return parser.parse_args()


def should_skip(relative_path: str, args: argparse.Namespace) -> Optional[str]:
    """
    Decide whether a file is a training byproduct rather than a result.

    Args:
        relative_path: path of the file relative to the repository root, with '/' separators.
        args: the parsed command line, whose ``--include_*`` flags relax the rules.

    Returns:
        The reason the file is skipped, or None if it belongs in the archive.
    """
    parts = relative_path.split('/')
    name = parts[-1]
    if not args.keep_all_checkpoints and 'ModelParameters' in parts \
            and CHECKPOINT_SNAPSHOT.match(name):
        return 'periodic checkpoint snapshot'
    if not args.include_optimizer and any(fnmatch.fnmatch(p, '*Optimizer*') for p in parts):
        return 'optimizer state'
    if not args.include_plots and 'plots' in parts:
        return 'training visualisation'
    if not args.include_tb and name.startswith('events.out.tfevents'):
        return 'tensorboard event file'
    return None


def collect(root: str, prefix: str, args: argparse.Namespace) -> Tuple[List[Tuple[str, str]],
                                                                      Dict[str, Tuple[int, int]]]:
    """
    Walk a directory and split its files into "pack this" and "skipped".

    Args:
        root: directory to walk.
        prefix: path inside the archive that ``root`` maps to.
        args: the parsed command line.

    Returns:
        entries: ``(absolute path, path inside the archive)`` for every file to pack.
        skipped: reason -> (file count, total bytes) for the files left out.
    """
    entries: List[Tuple[str, str]] = []
    skipped: Dict[str, Tuple[int, int]] = {}
    for directory, _, files in os.walk(root):
        for name in sorted(files):
            path = os.path.join(directory, name)
            if os.path.islink(path) or not os.path.isfile(path):
                continue
            arcname = os.path.join(prefix, os.path.relpath(path, root)).replace(os.sep, '/')
            reason = should_skip(arcname, args)
            if reason is None:
                entries.append((path, arcname))
            else:
                count, size = skipped.get(reason, (0, 0))
                skipped[reason] = (count + 1, size + os.path.getsize(path))
    entries.sort(key=lambda item: item[1])
    return entries, skipped


def human(size: float) -> str:
    """
    Format a byte count.

    Args:
        size: number of bytes.

    Returns:
        The size with a binary unit suffix, e.g. ``'1.4 GiB'``.
    """
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB']:
        if size < 1024.0 or unit == 'TiB':
            return '{:.1f} {}'.format(size, unit)
        size /= 1024.0
    return '{:.1f} TiB'.format(size)


def git_commit(repo: str) -> str:
    """
    Read the commit the results were produced at, for provenance.

    Args:
        repo: repository directory.

    Returns:
        The short commit hash, or ``'unknown'`` outside a git checkout.
    """
    try:
        out = subprocess.run(['git', '-C', repo, 'rev-parse', '--short', 'HEAD'],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or 'unknown'
    except Exception:
        return 'unknown'


def object_rows(entries: List[Tuple[str, str]]) -> List[Tuple[str, int, int]]:
    """
    Summarise the archive per top level run directory.

    Args:
        entries: the ``(path, arcname)`` pairs that will be packed.

    Returns:
        ``(run name, file count, total bytes)`` sorted by run name.
    """
    totals: Dict[str, Tuple[int, int]] = {}
    for path, arcname in entries:
        parts = arcname.split('/')
        run = parts[1] if len(parts) > 2 and parts[0] == 'exps' else parts[0]
        count, size = totals.get(run, (0, 0))
        totals[run] = (count + 1, size + os.path.getsize(path))
    return [(run, count, size) for run, (count, size) in sorted(totals.items())]


def build_manifest(entries: List[Tuple[str, str]], skipped: Dict[str, Tuple[int, int]],
                   args: argparse.Namespace, repo: str) -> str:
    """
    Describe the archive: what is in it, what was left out, and how to use it.

    Args:
        entries: the ``(path, arcname)`` pairs that will be packed.
        skipped: reason -> (count, bytes) of the files left out.
        args: the parsed command line.
        repo: repository directory, for the commit hash.

    Returns:
        The markdown text of ``MANIFEST.md``.
    """
    total = sum(os.path.getsize(path) for path, _ in entries)
    lines = [
        '# Structure from Duplicates -- trained runs and evaluation output',
        '',
        'Produced by `scripts/pack_results.py` from the `exps/` tree of one full sweep over the 15',
        '`DuplicateSingleImage` objects (`cmd_train.sh` then `cmd_eval.sh`).',
        '',
        '| | |',
        '| --- | --- |',
        '| files | {} |'.format(len(entries)),
        '| uncompressed | {} |'.format(human(total)),
        '| repository commit | `{}` |'.format(git_commit(repo)),
        '',
        '## What is in here',
        '',
        'Unzip in the repository root; every path lands where `exp_runner.py` expects it.',
        '',
        '| path | what |',
        '| --- | --- |',
        '| `exps/{Geo,Vis,Mat}-<object>/<timestamp>/checkpoints/ModelParameters/latest.pth` | '
        'the trained weights of each stage |',
        '| `exps/Mat-<object>-eval/<timestamp>/evals_value/` | 2D metrics (PSNR/SSIM/LPIPS for rgb '
        'and albedo, normal/roughness error) |',
        '| `exps/Mat-<object>-eval/<timestamp>/evals_image/` | the rendered comparisons |',
        '| `exps/Mat-<object>-mesh/<timestamp>/mesh/` | the exported mesh (`mesh.ply`, '
        '`mesh_world.ply`, `mesh_attributes.npz`), the estimated envmap, and `transforms.json` |',
        '| `exps/Mat-<object>-mesh/<timestamp>/mesh/metrics_3d_{local,world}.json` | 3D metrics '
        '(Chamfer, F-score, normal consistency) -- synthetic objects only |',
        '| `exps/*/<timestamp>/setting.yaml` | the config each run was trained with |',
        '| `train_logs/<object>.log`, `<object>_eval.log` | training and eval logs |',
        '',
        '## What was left out, and what that costs you',
        '',
        '| left out | files | bytes |',
        '| --- | --- | --- |',
    ]
    for reason, (count, size) in sorted(skipped.items()):
        lines.append('| {} | {} | {} |'.format(reason, count, human(size)))
    lines += [
        '',
        '* **Optimizer state.** `--is_continue` *training* needs it, so training cannot be resumed',
        '  from this archive. Everything that only loads weights works: `--eval`, `--eval_relight`,',
        '  `--to_mesh`, `--to_uv`, and using a `Geo`/`Vis` checkpoint to start the next stage.',
        '* **Periodic checkpoints.** Only the final `latest.pth` of each stage is kept, so an',
        '  earlier iteration cannot be inspected.',
        '* **`plots/` and tensorboard events.** Training-time visualisations only; no metric is',
        '  computed from them.',
        '',
        '## Reading the numbers',
        '',
        'The six real-world objects (`airplane cake cheese cola potato yogurt`) have no albedo /',
        'roughness / normal ground truth and no Blender scene, so their albedo and roughness',
        'metrics and all 3D metrics are absent by construction, not missing by accident.',
        '',
        'For the 3D metrics quote `metrics_3d_world.json` and the `*_relative` fields (a fraction of',
        'the ground-truth bounding box diagonal), since the objects differ in size by more than an',
        'order of magnitude. Read them together with `pose_corner_spread`: divided by `diagonal` it',
        'is 0.64-1.09 % (mean 0.72 %) across the nine synthetic objects, the same order as the mean',
        'Chamfer-L1 (0.68 %), so these distances say about as much about the SfM poses as about the',
        'reconstructed shape. It is an upper bound on the pose contribution rather than a hard floor:',
        'it is measured at the corners of the canonical unit cube, which is wider than the objects.',
        '',
        'Where an experiment has more than one timestamped run, all of them are kept and the newest',
        'is the one the report quotes. The six real-world objects each have two `-eval` runs: the',
        'earlier one stopped after writing `evals_value/` (it hit the `align_scale` crash that commit',
        '`31e06d9` fixes), so take the later timestamp -- the one that also has `evals_image/`.',
        '',
        '## Per run',
        '',
        '| run | files | bytes |',
        '| --- | --- | --- |',
    ]
    for run, count, size in object_rows(entries):
        lines.append('| `{}` | {} | {} |'.format(run, count, human(size)))
    lines.append('')
    return '\n'.join(lines)


def write_zip(output: str, entries: List[Tuple[str, str]], manifest: str) -> None:
    """
    Write the archive, storing already-compressed payloads and deflating the rest.

    Args:
        output: path of the zip to write.
        entries: the ``(path, arcname)`` pairs to pack.
        manifest: text of ``MANIFEST.md``, added at the archive root.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output)) or '.', exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6,
                         allowZip64=True) as archive:
        archive.writestr('MANIFEST.md', manifest)
        for index, (path, arcname) in enumerate(entries, start=1):
            stored = arcname.lower().endswith(STORED_SUFFIXES)
            archive.write(path, arcname,
                          compress_type=zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED)
            if index % 200 == 0 or index == len(entries):
                print('  {}/{} files'.format(index, len(entries)), flush=True)


def main() -> None:
    """Collect the results, describe them in a manifest, and write the zip."""
    args = parse_args()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    entries, skipped = collect(args.exps_dir, 'exps', args)
    if args.log_dir and os.path.isdir(args.log_dir):
        log_entries, log_skipped = collect(args.log_dir, 'train_logs', args)
        entries += log_entries
        for reason, (count, size) in log_skipped.items():
            have_count, have_size = skipped.get(reason, (0, 0))
            skipped[reason] = (have_count + count, have_size + size)

    total = sum(os.path.getsize(path) for path, _ in entries)
    print('packing {} files, {} uncompressed'.format(len(entries), human(total)))
    for reason, (count, size) in sorted(skipped.items()):
        print('  leaving out {:>5} files ({:>9}): {}'.format(count, human(size), reason))

    manifest = build_manifest(entries, skipped, args, repo)
    if args.dry_run:
        print('\n--dry_run, not writing {}\n'.format(args.output))
        print(manifest)
        return

    write_zip(args.output, entries, manifest)
    print('\nwrote {} ({})'.format(args.output, human(os.path.getsize(args.output))))


if __name__ == '__main__':
    main()
