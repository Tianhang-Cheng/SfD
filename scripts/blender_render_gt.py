"""
Render the Blender ground truth of a processed object from the training viewpoint.

Must be run by Blender, which is the only thing that can open a ``.blend``:

    blender --background blender_data/coffee/coffee_clean.blend \\
        --python scripts/blender_render_gt.py -- \\
        --data_split_dir hf_data/train_split/coffee --envmap envmaps/c.exr \\
        --output /tmp/coffee_gt.exr

Everything after ``--`` is parsed by this script. The camera is taken from
``blender_camera_gt_pose.json`` inside ``--data_split_dir``, i.e. exactly the camera the
released ``train/000_rgb.exr`` was rendered with: ``transform_matrix`` is a camera-to-world
matrix in Blender's own convention, so it can be assigned to ``camera.matrix_world``
unchanged, and ``camera_angle_x`` is the horizontal field of view.

Five of the nine released scenes keep the whole pile joined into a single mesh and the other four
hold one object per instance, but *both* already sit exactly as the training image was rendered,
so the scene is rendered untouched by default. ``--reapply_gt_poses`` re-places the objects from
``blender_object_gt_pose.json`` as a check on the pose bookkeeping; see :func:`reapply_gt_poses`.

Use ``scripts/compare_render.py`` afterwards to compare the result with the training image,
and ``scripts/check_blender_alignment.py`` to verify the pose bookkeeping without Blender.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from blender_common import JOINED, resolve_instances


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse the arguments that follow ``--`` on Blender's command line.

    Args:
        argv: argument list to parse; taken from ``sys.argv`` after ``--`` by default.

    Returns:
        The parsed arguments.
    """
    if argv is None:
        argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--blend_file', type=str, default='',
                        help='.blend to open; only needed when this runs through the bpy pip '
                             'module, since "blender --background <file>.blend" already opened it')
    parser.add_argument('--data_split_dir', type=str, required=True,
                        help='processed object directory holding blender_camera_gt_pose.json')
    parser.add_argument('--output', type=str, required=True,
                        help='destination image; .exr keeps the linear radiance')
    parser.add_argument('--envmap', type=str, default='',
                        help='HDRI used as the environment light (envmaps/c.exr for the '
                             'released renders); keeps the world of the .blend if empty')
    parser.add_argument('--envmap_strength', type=float, default=1.0,
                        help='multiplier on the environment light')
    parser.add_argument('--envmap_rotation_z', type=float, default=0.0,
                        help='rotate the HDRI around the up axis, in degrees, in case the '
                             'shading comes out rotated with respect to the training image')
    parser.add_argument('--resolution', type=int, default=800,
                        help='side length of the square render')
    parser.add_argument('--samples', type=int, default=128,
                        help='Cycles samples per pixel')
    parser.add_argument('--engine', type=str, default='CYCLES', choices=['CYCLES', 'BLENDER_EEVEE'],
                        help='render engine')
    parser.add_argument('--device', type=str, default='GPU', choices=['GPU', 'CPU'],
                        help='Cycles device')
    parser.add_argument('--film_transparent', default=True,
                        action=argparse.BooleanOptionalAction,
                        help='keep the background transparent, like the released renders')
    parser.add_argument('--reapply_gt_poses', default=False, action='store_true',
                        help='move the instance objects back onto the poses of '
                             'blender_object_gt_pose.json before rendering; not needed -- the '
                             'saved scenes already match the training images -- but a good check '
                             'that the recorded poses are right, since a correct pose file leaves '
                             'the render unchanged. Only works on the four scenes that keep one '
                             'object per instance')
    parser.add_argument('--frame', type=int, default=0,
                        help='index of the frame in blender_camera_gt_pose.json to render')
    return parser.parse_args(argv)


def load_camera(data_split_dir: str, frame: int) -> Dict[str, Any]:
    """
    Read the ground truth camera of one frame.

    Args:
        data_split_dir: processed object directory.
        frame: index into the ``frames`` list.

    Returns:
        A dict with ``matrix`` (4x4 nested list, camera-to-world) and ``angle_x`` in radians.
    """
    path = os.path.join(data_split_dir, 'blender_camera_gt_pose.json')
    with open(path, 'r') as handle:
        meta = json.load(handle)
    return {'matrix': meta['frames'][frame]['transform_matrix'],
            'angle_x': float(meta['camera_angle_x'])}


def setup_camera(matrix: List[List[float]], angle_x: float, resolution: int) -> Any:
    """
    Create the camera the dataset was rendered with and make it the active one.

    Args:
        matrix: 4x4 camera-to-world matrix in Blender convention.
        angle_x: horizontal field of view in radians.
        resolution: side length of the square render.

    Returns:
        The Blender camera object.
    """
    import bpy
    from mathutils import Matrix

    camera_data = bpy.data.cameras.new('sfd_gt_camera')
    camera_data.sensor_fit = 'HORIZONTAL'
    camera_data.angle_x = angle_x
    camera = bpy.data.objects.new('sfd_gt_camera', camera_data)
    bpy.context.scene.collection.objects.link(camera)
    camera.matrix_world = Matrix(matrix)
    bpy.context.scene.camera = camera
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.pixel_aspect_x = 1.0
    bpy.context.scene.render.pixel_aspect_y = 1.0
    return camera


def setup_environment_light(envmap: str, strength: float, rotation_z: float) -> None:
    """
    Replace the world of the scene with an HDRI environment light.

    Args:
        envmap: path to the ``.exr`` HDRI.
        strength: multiplier on the light.
        rotation_z: rotation of the HDRI around the up axis, in degrees.
    """
    import bpy
    from math import radians

    world = bpy.data.worlds.new('sfd_world')
    world.use_nodes = True
    bpy.context.scene.world = world
    nodes, links = world.node_tree.nodes, world.node_tree.links
    nodes.clear()

    output = nodes.new('ShaderNodeOutputWorld')
    background = nodes.new('ShaderNodeBackground')
    background.inputs['Strength'].default_value = strength
    texture = nodes.new('ShaderNodeTexEnvironment')
    texture.image = bpy.data.images.load(os.path.abspath(envmap))
    mapping = nodes.new('ShaderNodeMapping')
    mapping.inputs['Rotation'].default_value[2] = radians(rotation_z)
    coordinate = nodes.new('ShaderNodeTexCoord')

    links.new(coordinate.outputs['Generated'], mapping.inputs['Vector'])
    links.new(mapping.outputs['Vector'], texture.inputs['Vector'])
    links.new(texture.outputs['Color'], background.inputs['Color'])
    links.new(background.outputs['Background'], output.inputs['Surface'])


def reapply_gt_poses(data_split_dir: str) -> int:
    """
    Move every instance object back onto the pose ``blender_object_gt_pose.json`` records.

    The released scenes already sit exactly as they were rendered -- rendering them untouched
    reproduces the training image to a 0.00 px shift and a 0.999 silhouette IoU on all nine
    objects, see the README -- so this is *not* needed to reproduce the ground truth. It exists to
    prove the pose bookkeeping: if the recorded matrices are right, re-applying them changes
    nothing, and the render stays identical.

    It only works on the one-object-per-instance layout. The other five scenes hold the whole pile
    joined into a single mesh, whose ``matrix_world`` is not any instance's pose, so there is
    nothing to move.

    Unregistered instances are moved too: the training image was rendered with every instance in
    the scene, whether SfM later managed to register it or not.

    Args:
        data_split_dir: processed object directory.

    Returns:
        The number of instances that were placed.

    Raises:
        SystemExit: if the scene keeps the pile joined into one mesh.
    """
    import bpy
    from mathutils import Matrix

    instances, layout = resolve_instances(data_split_dir, include_unregistered=True)
    if layout == JOINED:
        raise SystemExit('this .blend holds the whole pile joined into one mesh, so the ground '
                         'truth poses cannot be re-applied to it; drop --reapply_gt_poses, the '
                         'saved scene already matches the training image')
    for instance in instances:
        bpy.data.objects[instance['object']].matrix_world = \
            Matrix([list(row) for row in instance['matrix']])
    return len(instances)


def setup_render(output: str, engine: str, samples: int, device: str,
                 film_transparent: bool) -> None:
    """
    Configure engine, sampling and output format.

    ``Standard`` view transform and a raw EXR keep the render in the same linear space as the
    released ``train/000_rgb.exr``; Blender's default Filmic would tone map it.

    Args:
        output: destination image path; the extension selects the format.
        engine: ``'CYCLES'`` or ``'BLENDER_EEVEE'``.
        samples: samples per pixel for Cycles.
        device: ``'GPU'`` or ``'CPU'``.
        film_transparent: render the background as transparent.
    """
    import bpy

    scene = bpy.context.scene
    scene.render.engine = engine
    scene.render.film_transparent = film_transparent
    if engine == 'CYCLES':
        scene.cycles.samples = samples
        scene.cycles.device = device
        if device == 'GPU':
            preferences = bpy.context.preferences.addons['cycles'].preferences
            preferences.get_devices()
            for candidate in ('OPTIX', 'CUDA', 'HIP', 'METAL', 'NONE'):
                try:
                    preferences.compute_device_type = candidate
                    break
                except TypeError:
                    continue
            for hardware in preferences.devices:
                hardware.use = hardware.type != 'CPU'

    try:  # 4.x renamed the colour management options
        scene.view_settings.view_transform = 'Standard'
    except TypeError:
        scene.view_settings.view_transform = 'Raw'
    scene.view_settings.look = 'None'
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    scene.render.filepath = os.path.abspath(output)
    if output.lower().endswith('.exr'):
        scene.render.image_settings.file_format = 'OPEN_EXR'
        scene.render.image_settings.color_depth = '32'
    else:
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_depth = '8'
    scene.render.image_settings.color_mode = 'RGBA' if film_transparent else 'RGB'


def main() -> None:
    """Set the scene up from the ground truth metadata and render one image."""
    try:
        import bpy
    except ImportError:
        raise SystemExit('this needs Blender: either\n'
                         '  blender --background <object>.blend --python {} -- --help\n'
                         'or the pip module, which brings its own Blender:\n'
                         '  pip install bpy && python {} -- --blend_file <object>.blend --help'
                         .format(os.path.relpath(__file__), os.path.relpath(__file__)))
    args = parse_args()
    if args.blend_file:
        bpy.ops.wm.open_mainfile(filepath=os.path.abspath(args.blend_file))

    camera = load_camera(args.data_split_dir, args.frame)
    setup_camera(camera['matrix'], camera['angle_x'], args.resolution)
    print('camera angle_x = {:.6f} rad -> focal {:.2f} px at {}'.format(
        camera['angle_x'],
        0.5 * args.resolution / __import__('math').tan(0.5 * camera['angle_x']),
        args.resolution))

    if args.envmap:
        setup_environment_light(args.envmap, args.envmap_strength, args.envmap_rotation_z)
        print('environment light: {}'.format(args.envmap))
    if args.reapply_gt_poses:
        print('re-applied the ground truth poses of {} instances'.format(
            reapply_gt_poses(args.data_split_dir)))

    setup_render(args.output, args.engine, args.samples, args.device, args.film_transparent)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    bpy.ops.render.render(write_still=True)
    print('wrote {}'.format(args.output))


if __name__ == '__main__':
    main()
