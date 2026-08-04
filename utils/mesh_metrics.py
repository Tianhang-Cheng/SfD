"""
Metrics for comparing a reconstructed mesh with a ground truth mesh.

Everything here works on triangle soups (``vertices``, ``faces``) and is implemented with
numpy plus ``scipy.spatial.cKDTree``, so no differentiable renderer or CUDA extension is
needed. The distances are computed between *surface samples*, not between vertices: vertex
distances depend on how finely each mesh happens to be tessellated, which makes them
incomparable between a marching cubes mesh and an artist's mesh.

The reported numbers follow the usual convention in the single view reconstruction
literature:

* ``chamfer_l1`` -- mean of the two one sided mean distances (the "Chamfer-L1" of Occupancy
  Networks, i.e. an average distance, not the average of squares),
* ``chamfer_l2`` -- the same with squared distances,
* ``f_score`` -- harmonic mean of precision and recall at a distance threshold, which is the
  metric to quote when the two meshes may differ in coverage,
* ``normal_consistency`` -- mean absolute cosine between the normals of nearest surface
  points, which catches a mesh that has the right silhouette but the wrong surface detail.

Since SfM leaves the scale free, :func:`umeyama` and :func:`icp_with_scale` are provided to
put the reconstruction in the ground truth frame before measuring. Prefer the analytic
alignment of ``utils/blender_align.py`` and use ICP only as a refinement: ICP will happily
hide a real pose error.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np


def load_mesh(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a mesh from any format trimesh understands, or from a ``mesh_attributes.npz``.

    Args:
        path: ``.ply`` / ``.obj`` / ... file, or an ``.npz`` written by ``--to_mesh``.

    Returns:
        vertices: ``[n,3]`` float64 array.
        faces: ``[m,3]`` int64 array.

    Raises:
        ValueError: if the file holds a scene with no triangles.
    """
    if path.endswith('.npz'):
        data = np.load(path)
        return (np.asarray(data['vertices'], dtype=np.float64),
                np.asarray(data['faces'], dtype=np.int64))

    import trimesh

    mesh = trimesh.load(path, process=False, force='mesh')
    if not hasattr(mesh, 'faces') or len(mesh.faces) == 0:
        raise ValueError('{} contains no triangles'.format(path))
    return (np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int64))


def face_normals(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unnormalised and normalised triangle normals.

    Args:
        vertices: ``[n,3]`` array.
        faces: ``[m,3]`` int array.

    Returns:
        normals: ``[m,3]`` unit normals (zero for degenerate triangles).
        areas: ``[m]`` triangle areas.
    """
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    length = np.linalg.norm(cross, axis=1)
    normals = cross / np.where(length[:, None] < 1e-20, 1.0, length[:, None])
    return normals, 0.5 * length


def sample_mesh_surface(vertices: np.ndarray, faces: np.ndarray, count: int,
                        seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample points uniformly over the surface of a triangle mesh.

    Triangles are drawn with a probability proportional to their area and a point is placed in
    each drawn triangle with the usual square-root barycentric trick, which makes the samples
    uniform per unit area rather than per triangle.

    Args:
        vertices: ``[n,3]`` array.
        faces: ``[m,3]`` int array.
        count: number of samples.
        seed: seed of the random generator, so a metric is reproducible.

    Returns:
        points: ``[count,3]`` float64 samples.
        normals: ``[count,3]`` unit normal of the triangle each sample came from.

    Raises:
        ValueError: if the mesh has no triangle with a positive area.
    """
    normals, areas = face_normals(vertices, faces)
    total = float(areas.sum())
    if total <= 0.0:
        raise ValueError('the mesh has zero surface area')

    generator = np.random.default_rng(seed)
    probability = areas / total
    index = generator.choice(len(faces), size=count, p=probability)

    triangles = vertices[faces[index]]
    barycentric = generator.random((count, 2))
    root = np.sqrt(barycentric[:, :1])
    weights = np.concatenate([1.0 - root, root * (1.0 - barycentric[:, 1:]),
                              root * barycentric[:, 1:]], axis=1)
    points = (weights[:, :, None] * triangles).sum(axis=1)
    return points, normals[index]


def nearest_neighbours(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Distance from every source point to the closest target point, and its index.

    Args:
        source: ``[n,3]`` query points.
        target: ``[m,3]`` reference points.

    Returns:
        distance: ``[n]`` distances.
        index: ``[n]`` index of the closest target point.
    """
    from scipy.spatial import cKDTree

    distance, index = cKDTree(target).query(source, k=1, workers=-1)
    return np.asarray(distance, dtype=np.float64), np.asarray(index, dtype=np.int64)


def chamfer_distance(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """
    Symmetric Chamfer distances between two point sets.

    Args:
        prediction: ``[n,3]`` samples of the reconstructed surface.
        target: ``[m,3]`` samples of the ground truth surface.

    Returns:
        A dict with ``chamfer_l1`` (mean of the two mean distances), ``chamfer_l2`` (the same
        with squared distances), the two one sided means ``accuracy`` / ``completeness``, their
        medians, and the two sided ``hausdorff`` distance.
    """
    forward, _ = nearest_neighbours(prediction, target)
    backward, _ = nearest_neighbours(target, prediction)
    return {
        'chamfer_l1': 0.5 * float(forward.mean() + backward.mean()),
        'chamfer_l2': 0.5 * float((forward ** 2).mean() + (backward ** 2).mean()),
        'accuracy': float(forward.mean()),
        'completeness': float(backward.mean()),
        'accuracy_median': float(np.median(forward)),
        'completeness_median': float(np.median(backward)),
        'hausdorff': float(max(forward.max(), backward.max())),
    }


def f_score(prediction: np.ndarray, target: np.ndarray, threshold: float) -> Dict[str, float]:
    """
    Precision, recall and F-score of a point set at a distance threshold.

    Args:
        prediction: ``[n,3]`` samples of the reconstructed surface.
        target: ``[m,3]`` samples of the ground truth surface.
        threshold: distance below which a sample counts as matched, in the units of the points.

    Returns:
        A dict with ``precision``, ``recall``, ``f_score`` and the ``threshold`` used.
    """
    forward, _ = nearest_neighbours(prediction, target)
    backward, _ = nearest_neighbours(target, prediction)
    precision = float((forward < threshold).mean())
    recall = float((backward < threshold).mean())
    denominator = precision + recall
    return {'precision': precision, 'recall': recall, 'threshold': float(threshold),
            'f_score': 0.0 if denominator <= 0.0 else 2.0 * precision * recall / denominator}


def normal_consistency(prediction: np.ndarray, prediction_normals: np.ndarray,
                       target: np.ndarray, target_normals: np.ndarray) -> Dict[str, float]:
    """
    Agreement of the surface normals of nearest neighbour samples.

    The absolute cosine is used, so a mesh whose faces are wound the other way still scores 1.

    Args:
        prediction: ``[n,3]`` samples of the reconstructed surface.
        prediction_normals: ``[n,3]`` unit normals of those samples.
        target: ``[m,3]`` samples of the ground truth surface.
        target_normals: ``[m,3]`` unit normals of those samples.

    Returns:
        A dict with ``normal_consistency`` in ``[0,1]`` and the two one sided values.
    """
    _, forward_index = nearest_neighbours(prediction, target)
    _, backward_index = nearest_neighbours(target, prediction)
    forward = np.abs((prediction_normals * target_normals[forward_index]).sum(axis=1))
    backward = np.abs((target_normals * prediction_normals[backward_index]).sum(axis=1))
    return {'normal_consistency': 0.5 * float(forward.mean() + backward.mean()),
            'normal_consistency_prediction': float(forward.mean()),
            'normal_consistency_target': float(backward.mean())}


def umeyama(source: np.ndarray, target: np.ndarray,
            with_scale: bool = True) -> Tuple[np.ndarray, float]:
    """
    Least squares similarity that maps ``source`` onto ``target`` for known correspondences.

    Args:
        source: ``[n,3]`` points.
        target: ``[n,3]`` corresponding points.
        with_scale: solve for a uniform scale as well as a rigid motion.

    Returns:
        matrix: ``[4,4]`` similarity such that ``matrix @ source ~ target``.
        rmse: the residual root mean squared error after applying it.
    """
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    centred_source = source - source_mean
    centred_target = target - target_mean

    covariance = centred_target.T @ centred_source / len(source)
    u, singular, vh = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vh) < 0:  # never mirror the object
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vh
    variance = float((centred_source ** 2).sum() / len(source))
    scale = float((singular * sign).sum() / variance) if with_scale and variance > 0 else 1.0

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = target_mean - scale * rotation @ source_mean
    residual = (source @ matrix[:3, :3].T + matrix[:3, 3][None]) - target
    return matrix, float(np.sqrt((residual ** 2).sum(axis=1).mean()))


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Apply a 4x4 transform to a point set.

    Args:
        points: ``[n,3]`` array.
        matrix: ``[4,4]`` transform.

    Returns:
        ``[n,3]`` transformed points.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    return points @ matrix[:3, :3].T + matrix[:3, 3][None]


def icp_with_scale(source: np.ndarray, target: np.ndarray, iterations: int = 50,
                   with_scale: bool = True, trim: float = 0.9,
                   initial: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    """
    Refine an alignment by iterated closest points, optionally solving for scale.

    Correspondences are the nearest target point of each source point; the worst ``1 - trim``
    of them are dropped each round so that a partially reconstructed surface does not drag the
    alignment.

    Args:
        source: ``[n,3]`` samples of the mesh to move.
        target: ``[m,3]`` samples of the fixed mesh.
        iterations: maximum number of rounds.
        with_scale: also solve for a uniform scale.
        trim: fraction of the best correspondences to keep, in ``(0,1]``.
        initial: ``[4,4]`` starting transform, identity by default.

    Returns:
        matrix: ``[4,4]`` transform mapping the source into the target frame.
        rmse: root mean squared nearest neighbour distance of the kept correspondences.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(target)
    matrix = np.eye(4, dtype=np.float64) if initial is None else np.asarray(initial,
                                                                           dtype=np.float64)
    keep = max(1, int(round(trim * len(source))))
    rmse = float('inf')
    for _ in range(max(iterations, 1)):
        moved = transform_points(source, matrix)
        distance, index = tree.query(moved, k=1, workers=-1)
        order = np.argsort(distance)[:keep]
        update, _ = umeyama(moved[order], target[index[order]], with_scale=with_scale)
        matrix = update @ matrix
        previous, rmse = rmse, float(np.sqrt((distance[order] ** 2).mean()))
        if abs(previous - rmse) < 1e-9 * max(rmse, 1e-9):
            break
    return matrix, rmse


def bounding_box_diagonal(points: np.ndarray) -> float:
    """
    Diagonal of the axis aligned bounding box of a point set.

    Args:
        points: ``[n,3]`` array.

    Returns:
        The length of the diagonal.
    """
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


def evaluate_meshes(prediction: Tuple[np.ndarray, np.ndarray],
                    target: Tuple[np.ndarray, np.ndarray],
                    samples: int = 200000,
                    f_score_thresholds: Tuple[float, ...] = (0.005, 0.01, 0.02),
                    seed: int = 0) -> Dict[str, Any]:
    """
    Full metric report for a reconstructed mesh against a ground truth mesh.

    Both meshes must already be in the same frame and the same units. Every distance is
    reported twice: in the units of the ground truth mesh, and divided by the ground truth
    bounding box diagonal, which is what makes numbers comparable across objects.

    Sampling two independent point sets on the *same* surface already gives a non zero Chamfer
    distance, so the numbers have a noise floor of roughly ``0.5 * sqrt(area / samples)``,
    about 0.11% of the bounding box diagonal at the default 200k samples, and it halves for
    every four fold increase. Raise ``samples`` before reading anything into a difference of
    that size.

    Args:
        prediction: ``(vertices, faces)`` of the reconstruction.
        target: ``(vertices, faces)`` of the ground truth.
        samples: number of surface samples drawn from each mesh.
        f_score_thresholds: thresholds for the F-score, as a fraction of the ground truth
            bounding box diagonal.
        seed: seed of the sampler.

    Returns:
        A dict with ``chamfer_l1``, ``chamfer_l2``, ``accuracy``, ``completeness``,
        ``hausdorff``, their ``*_relative`` versions, ``normal_consistency``,
        ``f_score@<threshold>`` entries, ``diagonal`` and ``samples``.
    """
    prediction_points, prediction_normals = sample_mesh_surface(prediction[0], prediction[1],
                                                                samples, seed=seed)
    target_points, target_normals = sample_mesh_surface(target[0], target[1], samples,
                                                        seed=seed + 1)
    diagonal = bounding_box_diagonal(target[0])

    report: Dict[str, Any] = dict(chamfer_distance(prediction_points, target_points))
    report.update(normal_consistency(prediction_points, prediction_normals,
                                     target_points, target_normals))
    for key in ['chamfer_l1', 'accuracy', 'completeness', 'accuracy_median',
                'completeness_median', 'hausdorff']:
        report[key + '_relative'] = report[key] / diagonal
    report['chamfer_l2_relative'] = report['chamfer_l2'] / diagonal ** 2

    for fraction in f_score_thresholds:
        scores = f_score(prediction_points, target_points, fraction * diagonal)
        report['f_score@{:g}'.format(fraction)] = scores['f_score']
        report['precision@{:g}'.format(fraction)] = scores['precision']
        report['recall@{:g}'.format(fraction)] = scores['recall']

    report['diagonal'] = diagonal
    report['samples'] = int(samples)
    return report
