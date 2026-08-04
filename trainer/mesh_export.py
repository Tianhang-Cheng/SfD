"""
Export the trained networks as a mesh (``--to_mesh``) or as a textured UV asset
(``--to_uv``).

The two entry points are used by every train runner, so the export behaves the same
whether it is run after stage ``Geo`` (geometry only) or after stage ``Mat`` (geometry plus
spatially varying BRDF and the estimated environment light).

All geometry is extracted in the canonical space of the shared object -- the space the SDF
network is queried in. ``transforms.json`` is written next to the mesh with everything
needed to move it into the SfM world frame (and from there into the Blender frame, see
``utils/blender_align.py``).
"""

import json
import os
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
from termcolor import colored

from model.material_sg import compute_envmap
from utils import mesh_util


def build_sdf_query(model: torch.nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Wrap the SDF network so that it can be called on raw canonical points.

    Args:
        model: the ``DupNeuSRenderer``.

    Returns:
        A callable mapping points ``[n,3]`` to signed distances ``[n]``.
    """
    def sdf_query(points: torch.Tensor) -> torch.Tensor:
        """Signed distance of canonical points (``transform_func=None`` skips the pose)."""
        return model.sdf_network.sdf(points, transform_func=None)
    return sdf_query


def build_gradient_query(model: torch.nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Wrap the analytic SDF gradient so that it can be called on raw canonical points.

    Args:
        model: the ``DupNeuSRenderer``.

    Returns:
        A callable mapping points ``[n,3]`` to gradients ``[n,3]`` in canonical space.
    """
    def gradient_query(points: torch.Tensor) -> torch.Tensor:
        """Gradient of the canonical SDF, i.e. the outward surface normal direction."""
        return model.sdf_network.gradient(points, transform_func=None)['gradients_world']
    return gradient_query


def build_material_query(model: torch.nn.Module) -> Callable[[np.ndarray], Dict[str, np.ndarray]]:
    """
    Wrap the BRDF network as a batched numpy callable returning the PBR channels.

    Args:
        model: the ``DupNeuSRenderer``.

    Returns:
        A callable mapping points ``[n,3]`` (numpy, canonical space) to a dict with
        ``albedo`` ``[n,3]``, ``roughness`` ``[n,1]`` and ``metallic`` ``[n,1]``.
    """
    def network_query(points: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Evaluate the BRDF network and keep only the per point outputs we export."""
        material = model.envmap_material_network(points)
        return {'albedo': material['sg_diffuse_albedo'],
                'roughness': material['sg_roughness'],
                'metallic': material['sg_metallic']}

    def material_query(points: np.ndarray) -> Dict[str, np.ndarray]:
        """Batched numpy front end of ``network_query``."""
        return mesh_util.query_points_in_batches(network_query, points)
    return material_query


def save_transforms(model: torch.nn.Module, output_dir: str, extra: Dict[str, Any]) -> str:
    """
    Dump every transform needed to place the exported mesh back into the scene.

    Args:
        model: the ``DupNeuSRenderer``.
        output_dir: directory to write ``transforms.json`` into.
        extra: additional entries to store (marching cubes settings, stage name, ...).

    Returns:
        The path of the written json.
    """
    poses = model.obj_poses
    meta: Dict[str, Any] = dict(extra)
    meta['init_method'] = poses.init_method.name
    meta['same_obj_num'] = int(poses.same_obj_num)
    meta['visible_num'] = int(poses.visible_num)
    meta['non_empty_indexes'] = np.asarray(poses.non_empty_index).astype(int).tolist()
    # canonical (mesh) space -> world space of the scene, one matrix per visible instance
    meta['object_to_world'] = poses.get_pose(enable_scale=True).detach().cpu().numpy().tolist()
    meta['scale_matrix'] = (poses.scale_mat.detach().cpu().numpy().tolist()
                            if hasattr(poses, 'scale_mat') else None)
    meta['camera_to_world'] = poses.sfm_c.detach().cpu().numpy().tolist()
    if hasattr(poses, 'blender_c'):
        meta['blender_camera_to_world'] = poses.blender_c.detach().cpu().numpy().tolist()

    path = os.path.join(output_dir, 'transforms.json')
    with open(path, 'w') as handle:
        json.dump(meta, handle, indent=4, separators=(',', ': '))
    return path


def save_envmap(model: torch.nn.Module, output_dir: str,
                height: int = 256, width: int = 512) -> Optional[str]:
    """
    Write the estimated environment light as a latitude/longitude EXR, so that a renderer
    can reproduce the training illumination for the exported asset.

    Args:
        model: the ``DupNeuSRenderer``.
        output_dir: directory to write ``envmap.exr``/``envmap.png`` into.
        height: height of the equirectangular map.
        width: width of the equirectangular map.

    Returns:
        The path of the EXR, or None if the model has no material network.
    """
    import imageio

    material_network = getattr(model, 'envmap_material_network', None)
    if material_network is None:
        return None
    with torch.no_grad():
        envmap = compute_envmap(lgtSGs=material_network.get_light(), H=height, W=width,
                                upper_hemi=material_network.upper_hemi)
    envmap = envmap.detach().cpu().numpy().astype(np.float32)
    exr_path = os.path.join(output_dir, 'envmap.exr')
    imageio.imwrite(exr_path, envmap)
    mesh_util.save_texture_png(os.path.join(output_dir, 'envmap.png'),
                               np.clip(envmap, 0.0, 1.0) ** (1.0 / 2.2))
    return exr_path


def extract_geometry(model: torch.nn.Module,
                     resolution: int = 512,
                     bound: float = 1.0,
                     batch_size: int = 1 << 20,
                     keep_largest: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run marching cubes on the canonical SDF and compute per vertex normals.

    Args:
        model: the ``DupNeuSRenderer``.
        resolution: marching cubes grid resolution per axis.
        bound: half side length of the marched cube, in canonical units.
        batch_size: number of points per SDF call.
        keep_largest: keep only the largest connected component.

    Returns:
        vertices ``[n,3]``, faces ``[m,3]`` and outward vertex normals ``[n,3]``.
    """
    model.eval()
    # Vis/Mat/eval always read the geometry at full frequency, so the export does too.
    model.sdf_network.progress = 1.0

    vertices, faces = mesh_util.extract_mesh_from_sdf(
        build_sdf_query(model), resolution=resolution, bound=bound,
        batch_size=batch_size, keep_largest=keep_largest)
    normals = mesh_util.compute_sdf_normals(build_gradient_query(model), vertices)
    faces = mesh_util.orient_faces_outwards(vertices, faces, normals)
    return vertices, faces, normals


def export_mesh(model: torch.nn.Module,
                output_dir: str,
                stage: str,
                resolution: int = 512,
                bound: float = 1.0,
                batch_size: int = 1 << 20,
                keep_largest: bool = True,
                export_instances: bool = False,
                data_split_dir: str = '') -> Dict[str, str]:
    """
    Write the canonical mesh, its per vertex material and the transforms into ``output_dir``.

    Produced files:
        ``mesh.ply``            canonical mesh, vertex colours = diffuse albedo (sRGB)
        ``mesh_world.ply``      the same mesh placed at instance 0 in the SfM world frame
        ``mesh_instances.ply``  all visible instances placed in the SfM world frame (opt in)
        ``mesh_attributes.npz`` raw float arrays (vertices, faces, normals, BRDF channels)
        ``transforms.json``     canonical -> world / camera transforms
        ``envmap.exr/.png``     the estimated environment light (stage ``Mat`` only)

    Args:
        model: the ``DupNeuSRenderer``.
        output_dir: directory to write into; created if missing.
        stage: name of the training stage the checkpoint comes from ('Geo'/'Vis'/'Mat').
        resolution: marching cubes grid resolution per axis.
        bound: half side length of the marched cube, in canonical units.
        batch_size: number of points per SDF call.
        keep_largest: keep only the largest connected component.
        export_instances: also write every instance placed in the world frame.
        data_split_dir: dataset directory the checkpoint was trained on; recorded in
            ``transforms.json`` so the evaluation scripts can find the ground truth.

    Returns:
        A dict mapping a short name to each written file.
    """
    os.makedirs(output_dir, exist_ok=True)
    vertices, faces, normals = extract_geometry(model, resolution=resolution, bound=bound,
                                                batch_size=batch_size,
                                                keep_largest=keep_largest)

    attributes: Dict[str, np.ndarray] = {}
    colors = None
    if getattr(model, 'envmap_material_network', None) is not None and stage == 'Mat':
        print('  querying the BRDF network at {} vertices'.format(len(vertices)))
        attributes = build_material_query(model)(vertices)
        colors = mesh_util.linear_to_srgb(attributes['albedo'])

    written: Dict[str, str] = {}
    written['mesh'] = mesh_util.write_ply(os.path.join(output_dir, 'mesh.ply'),
                                         vertices, faces, colors=colors, normals=normals)

    object_to_world = model.obj_poses.get_pose(enable_scale=True).detach().cpu().numpy()
    written['mesh_world'] = mesh_util.write_ply(
        os.path.join(output_dir, 'mesh_world.ply'),
        mesh_util.transform_points(vertices, object_to_world[0]), faces, colors=colors,
        normals=mesh_util.transform_normals(normals, object_to_world[0]))

    if export_instances:
        placed = [(mesh_util.transform_points(vertices, pose), faces) for pose in object_to_world]
        all_vertices, all_faces = mesh_util.concatenate_meshes(placed)
        all_colors = None if colors is None else np.tile(colors, (len(object_to_world), 1))
        written['mesh_instances'] = mesh_util.write_ply(
            os.path.join(output_dir, 'mesh_instances.ply'), all_vertices, all_faces,
            colors=all_colors)

    npz_path = os.path.join(output_dir, 'mesh_attributes.npz')
    np.savez_compressed(npz_path, vertices=vertices, faces=faces, normals=normals,
                        **{key: value for key, value in attributes.items()})
    written['attributes'] = npz_path

    if stage == 'Mat':
        envmap_path = save_envmap(model, output_dir)
        if envmap_path is not None:
            written['envmap'] = envmap_path

    written['transforms'] = save_transforms(model, output_dir, extra={
        'stage': stage, 'space': 'canonical', 'mesh_resolution': int(resolution),
        'mesh_bound': float(bound), 'num_vertices': int(len(vertices)),
        'num_faces': int(len(faces)), 'data_split_dir': data_split_dir})

    print(colored('Wrote the mesh to {}'.format(output_dir), 'green', attrs=['bold']))
    for name, path in written.items():
        print('  {:16s} {}'.format(name, path))
    return written


def export_uv(model: torch.nn.Module,
              output_dir: str,
              stage: str,
              resolution: int = 512,
              bound: float = 1.0,
              batch_size: int = 1 << 20,
              keep_largest: bool = True,
              texture_resolution: int = 1024,
              samples_per_texel: int = 4,
              dilate_iterations: int = 8,
              data_split_dir: str = '') -> Dict[str, str]:
    """
    Write a UV unwrapped OBJ with baked PBR textures into ``output_dir``.

    Every texel is baked by querying the BRDF network at the surface point that texel maps
    to, so the textures carry more detail than the mesh tessellation.

    Produced files:
        ``mesh.obj`` / ``mesh.mtl``  the unwrapped mesh and its metallic/roughness material
        ``albedo.png``               base colour, sRGB encoded
        ``roughness.png``           linear roughness
        ``metallic.png``            linear metallic
        ``mask.png``                texels covered by the atlas
        ``transforms.json``         canonical -> world / camera transforms
        ``envmap.exr/.png``         the estimated environment light (stage ``Mat`` only)

    Args:
        model: the ``DupNeuSRenderer``.
        output_dir: directory to write into; created if missing.
        stage: name of the training stage the checkpoint comes from ('Geo'/'Vis'/'Mat').
        resolution: marching cubes grid resolution per axis.
        bound: half side length of the marched cube, in canonical units.
        batch_size: number of points per SDF call.
        keep_largest: keep only the largest connected component.
        texture_resolution: side length of the baked textures.
        samples_per_texel: average number of surface samples per texel while baking.
        dilate_iterations: how far to grow the baked region into the atlas gutters.
        data_split_dir: dataset directory the checkpoint was trained on; recorded in
            ``transforms.json`` so the evaluation scripts can find the ground truth.

    Returns:
        A dict mapping a short name to each written file.

    Raises:
        RuntimeError: if the checkpoint has no trained BRDF network (stage ``Geo``/``Vis``).
    """
    if getattr(model, 'envmap_material_network', None) is None or stage != 'Mat':
        raise RuntimeError('--to_uv bakes the BRDF, so it needs a Mat checkpoint; use '
                           '--to_mesh for a geometry only export of stage {}'.format(stage))

    os.makedirs(output_dir, exist_ok=True)
    vertices, faces, normals = extract_geometry(model, resolution=resolution, bound=bound,
                                                batch_size=batch_size,
                                                keep_largest=keep_largest)

    uv_vertices, uv_faces, uvs, vertex_mapping = mesh_util.unwrap_mesh_uv(vertices, faces)
    uv_normals = normals[vertex_mapping]

    maps = mesh_util.bake_attributes_to_texture(
        uv_vertices, uv_faces, uvs, build_material_query(model),
        resolution=texture_resolution, samples_per_texel=samples_per_texel,
        dilate_iterations=dilate_iterations)

    written: Dict[str, str] = {}
    written['albedo'] = mesh_util.save_texture_png(os.path.join(output_dir, 'albedo.png'),
                                                  maps['albedo'], srgb=True)
    written['roughness'] = mesh_util.save_texture_png(os.path.join(output_dir, 'roughness.png'),
                                                     maps['roughness'])
    written['metallic'] = mesh_util.save_texture_png(os.path.join(output_dir, 'metallic.png'),
                                                     maps['metallic'])
    written['mask'] = mesh_util.save_texture_png(os.path.join(output_dir, 'mask.png'),
                                                maps['mask'])
    written['mtl'] = mesh_util.write_pbr_mtl(
        os.path.join(output_dir, 'mesh.mtl'), 'sfd_material',
        {'albedo': 'albedo.png', 'roughness': 'roughness.png', 'metallic': 'metallic.png'})
    written['obj'] = mesh_util.write_obj_with_uv(
        os.path.join(output_dir, 'mesh.obj'), uv_vertices, uv_faces, uvs, normals=uv_normals,
        material_name='sfd_material', material_library='mesh.mtl')

    envmap_path = save_envmap(model, output_dir)
    if envmap_path is not None:
        written['envmap'] = envmap_path

    written['transforms'] = save_transforms(model, output_dir, extra={
        'stage': stage, 'space': 'canonical', 'mesh_resolution': int(resolution),
        'mesh_bound': float(bound), 'num_vertices': int(len(uv_vertices)),
        'num_faces': int(len(uv_faces)), 'texture_resolution': int(texture_resolution),
        'data_split_dir': data_split_dir})

    print(colored('Wrote the textured mesh to {}'.format(output_dir), 'green', attrs=['bold']))
    for name, path in written.items():
        print('  {:16s} {}'.format(name, path))
    return written
