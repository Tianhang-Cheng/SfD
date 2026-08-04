"""
Turn the trained networks into an ordinary asset: a triangle mesh plus (optionally) a UV
atlas with baked PBR textures.

Everything in here works in the *canonical* (a.k.a. local / object) space of the shared
object, which is the space the SDF and the BRDF network are queried in. That space is the
world space of the scene mapped through ``object_scale_matrix.json``, so the object lives
roughly inside the unit sphere; see the "Coordinate System" section of the README.

The module deliberately has no dependency on the trainers: callers pass in plain callables
that evaluate the networks, which keeps it usable from a notebook or a standalone script.
"""

import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# both are optional, and only the UV path needs them
try:  # pragma: no cover - availability depends on the environment
    import xatlas
except ImportError:  # pragma: no cover
    xatlas = None
try:  # pragma: no cover
    import trimesh
except ImportError:  # pragma: no cover
    trimesh = None


SdfQuery = Callable[[torch.Tensor], torch.Tensor]
PointQuery = Callable[[np.ndarray], Dict[str, np.ndarray]]


def evaluate_sdf_grid(sdf_query: SdfQuery,
                      resolution: int,
                      bound: float,
                      center: Sequence[float] = (0.0, 0.0, 0.0),
                      batch_size: int = 1 << 20,
                      device: str = 'cuda',
                      verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample the SDF on a regular grid covering ``center +- bound``.

    Args:
        sdf_query: maps points ``[n,3]`` to signed distances ``[n]`` (negative inside).
        resolution: number of samples per axis.
        bound: half side length of the cube to march, in canonical units.
        center: centre of the cube, in canonical units.
        batch_size: number of points evaluated per network call.
        device: device the query expects its input on.
        verbose: print a one line progress summary.

    Returns:
        volume: ``[resolution,resolution,resolution]`` float32 array of SDF values.
        grid_min: ``[3]`` float32 coordinate of voxel ``(0,0,0)``.
    """
    center_np = np.asarray(center, dtype=np.float32)
    axis = np.linspace(-bound, bound, resolution, dtype=np.float32)
    volume = np.empty([resolution, resolution, resolution], dtype=np.float32)

    # iterate over x slabs so that a single batch never exceeds batch_size points
    rows_per_batch = max(1, batch_size // (resolution * resolution))
    grid_y, grid_z = np.meshgrid(axis, axis, indexing='ij')
    plane = np.stack([np.zeros_like(grid_y), grid_y, grid_z], axis=-1).reshape(-1, 3)
    plane_torch = torch.from_numpy(plane).to(device)

    with torch.no_grad():
        for start in range(0, resolution, rows_per_batch):
            stop = min(start + rows_per_batch, resolution)
            chunk = plane_torch[None].repeat(stop - start, 1, 1)  # [r,res*res,3]
            chunk[..., 0] = torch.from_numpy(axis[start:stop]).to(device)[:, None]
            chunk = chunk.reshape(-1, 3) + torch.from_numpy(center_np).to(device)
            values = sdf_query(chunk).reshape(stop - start, resolution, resolution)
            volume[start:stop] = values.detach().float().cpu().numpy()
            if verbose:
                print('\r  sdf grid {}/{}'.format(stop, resolution), end='', flush=True)
    if verbose:
        print('\r  sdf grid {res}/{res}, range [{lo:.3f}, {hi:.3f}]'.format(
            res=resolution, lo=float(volume.min()), hi=float(volume.max())))
    return volume, (center_np - bound).astype(np.float32)


def keep_largest_component(vertices: np.ndarray, faces: np.ndarray,
                           min_face_ratio: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Drop every connected component but the largest one (by face count).

    Marching an SDF that was only supervised inside the visual hull usually produces a few
    stray blobs far from the object; they ruin both the UV atlas and the 3D metrics.

    Args:
        vertices: ``[n,3]`` float array.
        faces: ``[m,3]`` int array.
        min_face_ratio: also keep any component holding at least this fraction of the faces
            of the largest one (0 keeps only the largest).

    Returns:
        The filtered ``(vertices, faces)``; the input is returned unchanged when trimesh is
        not installed.
    """
    if trimesh is None:
        print('  trimesh is not installed, skipping the connected component filter')
        return vertices, faces

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    components = mesh.split(only_watertight=False)
    if len(components) <= 1:
        return vertices, faces
    sizes = np.array([len(c.faces) for c in components])
    threshold = max(1, int(sizes.max() * min_face_ratio)) if min_face_ratio > 0 else sizes.max()
    kept = [c for c, s in zip(components, sizes) if s >= threshold]
    merged = trimesh.util.concatenate(kept)
    print('  connected components: {} -> {} (faces {} -> {})'.format(
        len(components), len(kept), len(mesh.faces), len(merged.faces)))
    return (np.asarray(merged.vertices, dtype=np.float32),
            np.asarray(merged.faces, dtype=np.int32))


def extract_mesh_from_sdf(sdf_query: SdfQuery,
                          resolution: int = 512,
                          bound: float = 1.0,
                          level: float = 0.0,
                          center: Sequence[float] = (0.0, 0.0, 0.0),
                          batch_size: int = 1 << 20,
                          keep_largest: bool = True,
                          device: str = 'cuda',
                          verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Marching cubes on the canonical SDF.

    Args:
        sdf_query: maps points ``[n,3]`` to signed distances ``[n]`` (negative inside).
        resolution: number of samples per axis of the marching cubes grid.
        bound: half side length of the cube to march, in canonical units.
        level: iso value to extract, normally 0.
        center: centre of the cube, in canonical units.
        batch_size: number of points evaluated per network call.
        keep_largest: throw away all but the largest connected component.
        device: device the query expects its input on.
        verbose: print progress.

    Returns:
        vertices: ``[n,3]`` float32 array in canonical space.
        faces: ``[m,3]`` int32 array, wound counter clockwise seen from outside.

    Raises:
        RuntimeError: if the iso surface does not intersect the sampled cube.
    """
    from skimage import measure  # local import: only this function needs scikit-image

    volume, grid_min = evaluate_sdf_grid(sdf_query, resolution=resolution, bound=bound,
                                         center=center, batch_size=batch_size,
                                         device=device, verbose=verbose)
    if volume.min() > level or volume.max() < level:
        raise RuntimeError(
            'the {} iso surface is not inside the marched cube (sdf range [{:.3f}, {:.3f}]); '
            'try a larger --mesh_bound or check that the checkpoint is trained'.format(
                level, float(volume.min()), float(volume.max())))

    spacing = 2.0 * bound / (resolution - 1)
    # 'descent' is what winds the triangles outwards for an SDF that is negative inside
    # (verified on an analytic sphere); orient_faces_outwards() is the safety net.
    vertices, faces, _, _ = measure.marching_cubes(volume, level=level,
                                                   spacing=(spacing, spacing, spacing),
                                                   gradient_direction='descent')
    vertices = (vertices + grid_min[None]).astype(np.float32)
    faces = faces.astype(np.int32)
    if verbose:
        print('  marching cubes: {} vertices, {} faces'.format(len(vertices), len(faces)))
    if keep_largest:
        vertices, faces = keep_largest_component(vertices, faces)
    return vertices, faces


def query_points_in_batches(query: Callable[[torch.Tensor], Dict[str, torch.Tensor]],
                            points: np.ndarray,
                            batch_size: int = 1 << 18,
                            device: str = 'cuda') -> Dict[str, np.ndarray]:
    """
    Evaluate a network that returns a dict of per point tensors, in chunks.

    Args:
        query: maps points ``[n,3]`` to a dict of tensors whose first dimension is ``n``.
        points: ``[n,3]`` array of query positions.
        batch_size: number of points per network call.
        device: device the query expects its input on.

    Returns:
        A dict with the same keys, each value a ``[n,...]`` float32 numpy array.
    """
    collected: Dict[str, List[np.ndarray]] = {}
    with torch.no_grad():
        for start in range(0, len(points), batch_size):
            chunk = torch.from_numpy(points[start:start + batch_size]).float().to(device)
            for key, value in query(chunk).items():
                if not torch.is_tensor(value) or value.shape[0] != chunk.shape[0]:
                    continue  # e.g. the shared light parameters, which are not per point
                collected.setdefault(key, []).append(value.detach().float().cpu().numpy())
    return {key: np.concatenate(value, axis=0) for key, value in collected.items()}


def compute_sdf_normals(gradient_query: SdfQuery,
                        vertices: np.ndarray,
                        batch_size: int = 1 << 16,
                        device: str = 'cuda') -> np.ndarray:
    """
    Per vertex normals taken from the analytic SDF gradient (smoother than face normals).

    Args:
        gradient_query: maps points ``[n,3]`` to gradients ``[n,3]``.
        vertices: ``[n,3]`` array of positions in canonical space.
        batch_size: number of points per network call.
        device: device the query expects its input on.

    Returns:
        ``[n,3]`` float32 array of unit normals pointing away from the surface.
    """
    normals = np.empty_like(vertices, dtype=np.float32)
    for start in range(0, len(vertices), batch_size):
        chunk = torch.from_numpy(vertices[start:start + batch_size]).float().to(device)
        chunk.requires_grad_(True)
        gradient = gradient_query(chunk)
        normals[start:start + batch_size] = gradient.detach().float().cpu().numpy()
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
    return normals.astype(np.float32)


def orient_faces_outwards(vertices: np.ndarray, faces: np.ndarray,
                          normals: np.ndarray) -> np.ndarray:
    """
    Flip the winding of the triangles whose geometric normal disagrees with the SDF normal.

    Args:
        vertices: ``[n,3]`` array.
        faces: ``[m,3]`` int array.
        normals: ``[n,3]`` outward vertex normals.

    Returns:
        ``[m,3]`` int32 array of faces, all wound consistently outwards.
    """
    triangles = vertices[faces]
    face_normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    reference = normals[faces].mean(axis=1)
    flip = (face_normal * reference).sum(axis=1) < 0
    faces = faces.copy()
    faces[flip] = faces[flip][:, ::-1]
    if flip.any():
        print('  flipped the winding of {}/{} faces'.format(int(flip.sum()), len(faces)))
    return faces.astype(np.int32)


def linear_to_srgb(color: np.ndarray) -> np.ndarray:
    """
    sRGB transfer function, so that albedo written into an 8 bit PNG looks right.

    Args:
        color: array of linear values, expected in ``[0,1]``.

    Returns:
        An array of the same shape holding the sRGB encoded values.
    """
    color = np.clip(color, 0.0, 1.0)
    return np.where(color <= 0.0031308, color * 12.92, 1.055 * color ** (1.0 / 2.4) - 0.055)


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Apply a 4x4 homogeneous transform to a point cloud.

    Args:
        points: ``[n,3]`` array.
        matrix: ``[4,4]`` array.

    Returns:
        ``[n,3]`` float32 array of transformed points.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    transformed = points @ matrix[:3, :3].T + matrix[:3, 3][None]
    return transformed.astype(np.float32)


def transform_normals(normals: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Apply the inverse transpose of a 4x4 transform to normals and renormalise.

    Args:
        normals: ``[n,3]`` array.
        matrix: ``[4,4]`` array.

    Returns:
        ``[n,3]`` float32 array of unit normals.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    transformed = normals @ np.linalg.inv(matrix[:3, :3])
    transformed = transformed / (np.linalg.norm(transformed, axis=1, keepdims=True) + 1e-12)
    return transformed.astype(np.float32)


def concatenate_meshes(meshes: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Concatenate ``(vertices, faces)`` pairs into a single mesh, offsetting the indices.

    Args:
        meshes: sequence of ``(vertices [n,3], faces [m,3])`` pairs.

    Returns:
        The merged ``(vertices, faces)``.
    """
    all_vertices, all_faces, offset = [], [], 0
    for vertices, faces in meshes:
        all_vertices.append(np.asarray(vertices, dtype=np.float32))
        all_faces.append(np.asarray(faces, dtype=np.int64) + offset)
        offset += len(vertices)
    return (np.concatenate(all_vertices, axis=0),
            np.concatenate(all_faces, axis=0).astype(np.int32))


def write_ply(path: str,
              vertices: np.ndarray,
              faces: np.ndarray,
              colors: Optional[np.ndarray] = None,
              normals: Optional[np.ndarray] = None) -> str:
    """
    Write a binary little endian PLY with optional vertex colours and normals.

    Args:
        path: destination file.
        vertices: ``[n,3]`` array.
        faces: ``[m,3]`` int array.
        colors: ``[n,3]`` array in ``[0,1]`` (written as uint8) or None.
        normals: ``[n,3]`` array or None.

    Returns:
        The path that was written.
    """
    vertices = np.asarray(vertices, dtype='<f4')
    faces = np.asarray(faces, dtype='<i4')

    fields = [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]
    columns = [vertices]
    header = ['ply', 'format binary_little_endian 1.0',
              'element vertex {}'.format(len(vertices)),
              'property float x', 'property float y', 'property float z']
    if normals is not None:
        fields += [('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4')]
        columns.append(np.asarray(normals, dtype='<f4'))
        header += ['property float nx', 'property float ny', 'property float nz']
    if colors is not None:
        fields += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        columns.append(np.clip(np.asarray(colors) * 255.0 + 0.5, 0, 255).astype('u1'))
        header += ['property uchar red', 'property uchar green', 'property uchar blue']
    header += ['element face {}'.format(len(faces)),
               'property list uchar int vertex_indices', 'end_header']

    vertex_data = np.empty(len(vertices), dtype=fields)
    cursor = 0
    for column in columns:
        for index in range(column.shape[1]):
            vertex_data[fields[cursor][0]] = column[:, index]
            cursor += 1

    face_data = np.empty(len(faces), dtype=[('n', 'u1'), ('i', '<i4', (3,))])
    face_data['n'] = 3
    face_data['i'] = faces

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'wb') as handle:
        handle.write(('\n'.join(header) + '\n').encode('ascii'))
        handle.write(vertex_data.tobytes())
        handle.write(face_data.tobytes())
    return path


def unwrap_mesh_uv(vertices: np.ndarray, faces: np.ndarray,
                   verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a UV atlas with xatlas.

    xatlas cuts the mesh along the atlas seams, so the returned mesh has more vertices than
    the input one and ``vertex_mapping`` says where each new vertex came from.

    Args:
        vertices: ``[n,3]`` array.
        faces: ``[m,3]`` int array.
        verbose: print the vertex/face count after the unwrap.

    Returns:
        vertices_uv: ``[k,3]`` positions of the cut mesh.
        faces_uv: ``[m,3]`` int32 faces indexing ``vertices_uv``.
        uvs: ``[k,2]`` float32 texture coordinates in ``[0,1]``.
        vertex_mapping: ``[k]`` int32 index into the original vertices.

    Raises:
        ImportError: if xatlas is not installed.
    """
    if xatlas is None:
        raise ImportError('--to_uv needs xatlas: pip install xatlas')
    vertex_mapping, faces_uv, uvs = xatlas.parametrize(
        np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.uint32))
    vertex_mapping = vertex_mapping.astype(np.int32)
    if verbose:
        print('  uv atlas: {} -> {} vertices, {} faces'.format(
            len(vertices), len(vertex_mapping), len(faces_uv)))
    return (np.asarray(vertices, dtype=np.float32)[vertex_mapping],
            faces_uv.astype(np.int32), uvs.astype(np.float32), vertex_mapping)


def bake_attributes_to_texture(vertices: np.ndarray,
                               faces: np.ndarray,
                               uvs: np.ndarray,
                               point_query: PointQuery,
                               resolution: int = 1024,
                               samples_per_texel: int = 4,
                               dilate_iterations: int = 8,
                               seed: int = 0,
                               verbose: bool = True) -> Dict[str, np.ndarray]:
    """
    Bake per point network outputs into texture maps through the UV atlas.

    Instead of interpolating vertex attributes, the surface point of every texel is
    reconstructed and the network is queried there, so the texture keeps detail the mesh
    tessellation cannot carry. Texels are covered by stratified random samples over the UV
    triangles and the holes left over (atlas gutters) are filled by dilation.

    Args:
        vertices: ``[n,3]`` positions of the unwrapped mesh, in canonical space.
        faces: ``[m,3]`` int faces indexing ``vertices`` and ``uvs``.
        uvs: ``[n,2]`` texture coordinates in ``[0,1]``.
        point_query: maps points ``[k,3]`` to a dict of ``[k,c]`` attributes to bake.
        resolution: side length of the square textures.
        samples_per_texel: average number of surface samples per covered texel.
        dilate_iterations: how many times to grow the baked region into empty texels.
        seed: seed of the sampler, for reproducible bakes.
        verbose: print progress.

    Returns:
        A dict with one ``[resolution,resolution,c]`` float32 map per attribute plus a
        ``mask`` entry, ``[resolution,resolution]`` float32, that is 1 on baked texels.
    """
    rng = np.random.default_rng(seed)
    uv_pixel = np.stack([uvs[:, 0] * resolution, (1.0 - uvs[:, 1]) * resolution], axis=-1)
    triangle_uv = uv_pixel[faces]                                        # [m,3,2]
    edge_a = triangle_uv[:, 1] - triangle_uv[:, 0]
    edge_b = triangle_uv[:, 2] - triangle_uv[:, 0]
    area = 0.5 * np.abs(edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0])

    counts = np.maximum(1, np.ceil(area * samples_per_texel)).astype(np.int64)
    triangle_index = np.repeat(np.arange(len(faces), dtype=np.int64), counts)
    if verbose:
        print('  baking {} samples into a {}x{} atlas'.format(len(triangle_index), resolution,
                                                              resolution))

    # uniform barycentric samples inside each triangle
    r1, r2 = rng.random(len(triangle_index)), rng.random(len(triangle_index))
    sqrt_r1 = np.sqrt(r1)
    bary = np.stack([1.0 - sqrt_r1, sqrt_r1 * (1.0 - r2), sqrt_r1 * r2], axis=-1)  # [k,3]

    sample_uv = (bary[..., None] * triangle_uv[triangle_index]).sum(axis=1)        # [k,2]
    column = np.clip(sample_uv[:, 0].astype(np.int64), 0, resolution - 1)
    row = np.clip(sample_uv[:, 1].astype(np.int64), 0, resolution - 1)
    flat_index = row * resolution + column

    sample_xyz = (bary[..., None] * vertices[faces][triangle_index]).sum(axis=1)   # [k,3]
    attributes = point_query(sample_xyz)

    weight = np.bincount(flat_index, minlength=resolution * resolution).astype(np.float32)
    coverage = (weight > 0).astype(np.float32).reshape(resolution, resolution)
    maps: Dict[str, np.ndarray] = {}
    for key, value in attributes.items():
        value = np.atleast_2d(np.asarray(value, dtype=np.float32))
        channels = value.shape[1]
        accumulated = np.stack([
            np.bincount(flat_index, weights=value[:, c].astype(np.float64),
                        minlength=resolution * resolution) for c in range(channels)], axis=-1)
        texture = (accumulated / np.maximum(weight, 1.0)[:, None]).astype(np.float32)
        maps[key] = texture.reshape(resolution, resolution, channels)

    if verbose:
        print('  atlas coverage before dilation: {:.1%}'.format(float(coverage.mean())))
    if dilate_iterations > 0:
        maps, _ = dilate_texture(maps, coverage, iterations=dilate_iterations)
    # the mask is the *baked* region: the dilated gutter around it is only there to stop the
    # background from bleeding in when the texture is filtered
    maps['mask'] = coverage
    return maps


def dilate_texture(maps: Dict[str, np.ndarray], mask: np.ndarray,
                   iterations: int = 8) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Grow baked texels into their empty neighbours, which hides the atlas seams.

    Args:
        maps: dict of ``[h,w,c]`` float32 textures.
        mask: ``[h,w]`` float32 coverage mask, 1 where a texel holds data.
        iterations: number of one pixel dilation steps.

    Returns:
        The dilated maps and the grown mask.
    """
    kernel = torch.ones(1, 1, 3, 3, dtype=torch.float32)
    mask_torch = torch.from_numpy(mask)[None, None]
    stacked = {key: torch.from_numpy(value).permute(2, 0, 1)[None] for key, value in maps.items()}

    for _ in range(iterations):
        if float(mask_torch.min()) > 0.5:
            break
        neighbours = torch.nn.functional.conv2d(mask_torch, kernel, padding=1)
        for key, value in stacked.items():
            channels = value.shape[1]
            summed = torch.nn.functional.conv2d(value * mask_torch,
                                                kernel.expand(channels, 1, 3, 3),
                                                padding=1, groups=channels)
            filled = summed / neighbours.clamp(min=1.0)
            stacked[key] = torch.where(mask_torch > 0.5, value, filled)
        mask_torch = ((mask_torch + (neighbours > 0).float()) > 0.5).float()

    return ({key: value[0].permute(1, 2, 0).numpy() for key, value in stacked.items()},
            mask_torch[0, 0].numpy())


def write_obj_with_uv(path: str,
                      vertices: np.ndarray,
                      faces: np.ndarray,
                      uvs: np.ndarray,
                      normals: Optional[np.ndarray] = None,
                      material_name: Optional[str] = None,
                      material_library: Optional[str] = None) -> str:
    """
    Write a Wavefront OBJ that references a material library.

    Args:
        path: destination ``.obj`` file.
        vertices: ``[n,3]`` array.
        faces: ``[m,3]`` int array indexing ``vertices``, ``uvs`` and ``normals``.
        uvs: ``[n,2]`` texture coordinates in ``[0,1]``.
        normals: ``[n,3]`` array or None.
        material_name: name of the material to use, or None for no material.
        material_library: file name of the ``.mtl`` to reference, or None.

    Returns:
        The path that was written.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    lines = ['# exported by SfD (utils/mesh_util.py)']
    if material_library is not None:
        lines.append('mtllib {}'.format(material_library))
    lines += ['v {:.6f} {:.6f} {:.6f}'.format(*v) for v in vertices]
    lines += ['vt {:.6f} {:.6f}'.format(*t) for t in uvs]
    if normals is not None:
        lines += ['vn {:.6f} {:.6f} {:.6f}'.format(*n) for n in normals]
    if material_name is not None:
        lines.append('usemtl {}'.format(material_name))
    one_based = np.asarray(faces, dtype=np.int64) + 1
    if normals is None:
        lines += ['f {0}/{0} {1}/{1} {2}/{2}'.format(*f) for f in one_based]
    else:
        lines += ['f {0}/{0}/{0} {1}/{1}/{1} {2}/{2}/{2}'.format(*f) for f in one_based]
    with open(path, 'w') as handle:
        handle.write('\n'.join(lines) + '\n')
    return path


def write_pbr_mtl(path: str, material_name: str, texture_names: Dict[str, str]) -> str:
    """
    Write a ``.mtl`` describing a metallic/roughness material.

    ``map_Pr``/``map_Pm`` are the PBR extension of the OBJ format that Blender's importer
    understands; a viewer that only knows plain OBJ still gets the base colour.

    Args:
        path: destination ``.mtl`` file.
        material_name: name referenced by ``usemtl`` in the OBJ.
        texture_names: maps ``'albedo'``, ``'roughness'``, ``'metallic'`` to image file
            names (relative to the ``.mtl``); missing keys are skipped.

    Returns:
        The path that was written.
    """
    lines = ['# exported by SfD (utils/mesh_util.py)', 'newmtl {}'.format(material_name),
             'Ka 0.000 0.000 0.000', 'Kd 1.000 1.000 1.000', 'Ks 0.000 0.000 0.000', 'd 1.0',
             'illum 2']
    if 'albedo' in texture_names:
        lines.append('map_Kd {}'.format(texture_names['albedo']))
    if 'roughness' in texture_names:
        lines.append('map_Pr {}'.format(texture_names['roughness']))
    if 'metallic' in texture_names:
        lines.append('map_Pm {}'.format(texture_names['metallic']))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as handle:
        handle.write('\n'.join(lines) + '\n')
    return path


def save_texture_png(path: str, texture: np.ndarray, srgb: bool = False) -> str:
    """
    Save a float texture as an 8 bit PNG.

    Args:
        path: destination ``.png`` file.
        texture: ``[h,w]`` or ``[h,w,c]`` float array in ``[0,1]``.
        srgb: apply the sRGB transfer function first (use it for albedo, not for
            roughness/metallic, which are linear data).

    Returns:
        The path that was written.
    """
    import imageio.v2 as imageio

    if texture.ndim == 3 and texture.shape[2] == 1:
        texture = texture[:, :, 0]
    if srgb:
        texture = linear_to_srgb(texture)
    image = np.clip(texture * 255.0 + 0.5, 0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    imageio.imwrite(path, image)
    return path
