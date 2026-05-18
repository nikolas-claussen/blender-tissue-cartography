"""
Visualization utilities: bounding boxes, ortho-slice planes, materials, and vertex shading.
"""

import bpy
import numpy as np

from .io_utils import normalize_quantiles


def create_box(length, width, height, name="RectangularBox", hide=True):
    """
    Create a rectangular box with one corner at the origin in the positive quadrant.

    Parameters
    ----------
    length : float
        Length along X.
    width : float
        Width along Y.
    height : float
        Height along Z.
    name : str, optional
        Name of the created object.
    hide : bool, optional
        Whether to hide the object in the viewport.

    Returns
    -------
    bpy.types.Object
    """
    current_active = bpy.context.active_object
    bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = (length / 2, width / 2, height / 2)
    obj.location = (length / 2, width / 2, height / 2)
    bpy.ops.object.transform_apply(location=True, scale=True)
    obj.hide_set(hide)
    if current_active:
        bpy.context.view_layer.objects.active = current_active
    return obj


def create_slice_plane(length, width, height, axis='z', position=0.0):
    """
    Create a 2D plane as a cross-section of a rectangular box along the specified axis.

    Parameters
    ----------
    length, width, height : float
        Box dimensions along X, Y, Z.
    axis : {'x', 'y', 'z'}
        Axis along which to slice.
    position : float
        Position along the chosen axis in the same units as the box dimensions.

    Returns
    -------
    bpy.types.Object
    """
    current_active = bpy.context.active_object
    if axis not in {'x', 'y', 'z'}:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")
    axis_limits = {'x': length, 'y': width, 'z': height}
    if not (0.0 <= position <= axis_limits[axis]):
        raise ValueError(
            f"Position must be within [0, {axis_limits[axis]}] for axis {axis}."
        )

    if axis == 'x':
        plane_size = (height, width)
        location = (position, width / 2, height / 2)
        rotation = (0, np.pi / 2, 0)
    elif axis == 'y':
        plane_size = (length, height)
        location = (length / 2, position, height / 2)
        rotation = (np.pi / 2, 0, 0)
    else:
        plane_size = (length, width)
        location = (length / 2, width / 2, position)
        rotation = (0, 0, 0)

    bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 0))
    plane = bpy.context.active_object
    plane.name = f"SlicePlane_{axis.upper()}_{position:.2f}"
    plane.scale = (plane_size[0] / 2, plane_size[1] / 2, 1)
    plane.location = location
    plane.rotation_euler = rotation
    bpy.ops.object.transform_apply(location=True, scale=True, rotation=True)

    if current_active:
        bpy.context.view_layer.objects.active = current_active
    return plane


def get_slice_image(image_3d, resolution, axis='z', position=0.0):
    """
    Extract an ortho-slice from a 4D image array.

    Parameters
    ----------
    image_3d : np.array of shape (n_channels, nx, ny, nz)
        Volumetric image.
    resolution : array-like of shape (3,)
        Voxel size in microns per axis.
    axis : {'x', 'y', 'z'}
        Slicing axis.
    position : float
        Position along the axis in microns.

    Returns
    -------
    np.array of shape (n_channels, h, w)
    """
    if axis == 'x':
        ind = np.clip(int(np.round(position / resolution[0])), 0, image_3d.shape[1] - 1)
        return image_3d[:, ind, :, ::-1]
    elif axis == 'y':
        ind = np.clip(int(np.round(position / resolution[1])), 0, image_3d.shape[2] - 1)
        return image_3d[:, :, ind, :].transpose((0, 2, 1))
    else:
        ind = np.clip(int(np.round(position / resolution[2])), 0, image_3d.shape[3] - 1)
        return image_3d[:, :, :, ind].transpose((0, 2, 1))


def create_material_from_array(slice_plane, array, material_name="SliceMaterial"):
    """
    Create and assign a material for a slice plane from a 2D or 3D numpy array.

    Parameters
    ----------
    slice_plane : bpy.types.Object
        The plane object to receive the material.
    array : np.array
        2D grayscale (H, W) or 3D RGBA (H, W, 4) array with values in [0, 1].
    material_name : str, optional
        Name for the new material.
    """
    if array.ndim not in (2, 3):
        raise ValueError("Input array must be 2D (grayscale) or 3D (RGBA).")

    image_height, image_width = array.shape[:2]
    pixel_data = np.zeros((image_height, image_width, 4), dtype=np.float32)
    if array.ndim == 2:
        pixel_data[..., 0] = pixel_data[..., 1] = pixel_data[..., 2] = array
        pixel_data[..., 3] = 1.0
    else:
        pixel_data[...] = array

    image = bpy.data.images.new(name="SliceTexture", width=image_width, height=image_height)
    image.pixels.foreach_set(pixel_data.flatten())

    material = bpy.data.materials.new(name=material_name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    for node in nodes:
        nodes.remove(node)

    texture_node = nodes.new(type="ShaderNodeTexImage")
    texture_node.image = image
    texture_node.location = (-400, 0)

    bsdf_node = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf_node.location = (0, 0)

    output_node = nodes.new(type="ShaderNodeOutputMaterial")
    output_node.location = (400, 0)

    links.new(texture_node.outputs["Color"], bsdf_node.inputs["Base Color"])
    links.new(bsdf_node.outputs["BSDF"], output_node.inputs["Surface"])

    slice_plane.active_material = material


def create_material_from_multilayer_array(mesh, array, material_name="ProjectedMaterial"):
    """
    Create and assign a material for a mesh using a multi-channel, multi-layer projection.

    All channel/layer textures are added as image texture nodes in the shader graph,
    but only channel 0, layer 0 is wired to the BSDF Base Color by default.
    Additional textures can be connected manually in the shader editor.

    Parameters
    ----------
    mesh : bpy.types.Object
        The mesh object to receive the material.
    array : np.array of shape (n_channels, n_layers, H, W)
        Projection data. Normalized per-channel before creating textures.
    material_name : str, optional
        Name for the new material.
    """
    if array.ndim != 4:
        raise ValueError("Input array must have 4 axes (channels, layers, H, W).")

    array_normalized = normalize_quantiles(array, quantiles=(0.01, 0.99), channel_axis=0,
                                           clip=True, data_type=None)
    image_height, image_width = array.shape[-2:]
    n_channels, n_layers = array.shape[:2]

    images = {}
    for ic, channel in enumerate(array_normalized):
        for il, layer in enumerate(channel):
            pixel_data = np.zeros((image_height, image_width, 4), dtype=np.float32)
            pixel_data[..., 0] = pixel_data[..., 1] = pixel_data[..., 2] = layer[::-1]
            pixel_data[..., 3] = 1.0
            img = bpy.data.images.new(
                name=f"Channel_{ic}_Layer_{il}",
                width=image_width, height=image_height,
            )
            img.pixels.foreach_set(pixel_data.flatten())
            images[(ic, il)] = img

    material = bpy.data.materials.new(name=material_name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    for node in nodes:
        nodes.remove(node)

    texture_nodes = {}
    for (ic, il), img in images.items():
        tex = nodes.new(type="ShaderNodeTexImage")
        tex.image = img
        tex.location = (-400, ic * 400 + il * 300)
        texture_nodes[(ic, il)] = tex

    bsdf_node = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf_node.location = (0, 0)
    output_node = nodes.new(type="ShaderNodeOutputMaterial")
    output_node.location = (400, 0)

    links.new(texture_nodes[(0, 0)].outputs["Color"], bsdf_node.inputs["Base Color"])
    links.new(bsdf_node.outputs["BSDF"], output_node.inputs["Surface"])

    mesh.active_material = material


def compute_edge_lengths(obj):
    """
    Compute lengths of all edges in a mesh object.

    Parameters
    ----------
    obj : bpy.types.Object
        Mesh object.

    Returns
    -------
    np.array of shape (n_edges,)
    """
    if obj.type != 'MESH':
        raise ValueError("The selected object is not a mesh.")
    bpy.context.view_layer.objects.active = obj
    if obj.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    n_verts = len(obj.data.vertices)
    n_edges = len(obj.data.edges)
    vert_co_flat = np.zeros(n_verts * 3, dtype=np.float32)
    obj.data.vertices.foreach_get("co", vert_co_flat)
    vert_co = vert_co_flat.reshape(n_verts, 3)
    edge_verts_flat = np.zeros(n_edges * 2, dtype=np.int32)
    obj.data.edges.foreach_get("vertices", edge_verts_flat)
    edge_verts = edge_verts_flat.reshape(n_edges, 2)
    return np.linalg.norm(vert_co[edge_verts[:, 0]] - vert_co[edge_verts[:, 1]], axis=1)


def assign_vertex_colors(obj, colors):
    """
    Assign RGB colors to each vertex of a mesh object.

    Parameters
    ----------
    obj : bpy.types.Object
        Mesh object.
    colors : array-like of shape (n_vertices, 3)
        RGB values in [0, 1] for each vertex.
    """
    if obj.type != 'MESH':
        raise ValueError("Object is not a mesh.")
    ca = obj.data.color_attributes
    if not ca:
        ca.new(name="Col", type='BYTE_COLOR', domain='CORNER')
    color_layer = ca.active_color
    n = len(obj.data.loops)
    loop_vert_flat = np.zeros(n, dtype=np.int32)
    obj.data.loops.foreach_get("vertex_index", loop_vert_flat)
    colors_rgba = np.ones((n, 4), dtype=np.float32)
    colors_rgba[:, :3] = np.asarray(colors, dtype=np.float32)[loop_vert_flat]
    color_layer.data.foreach_set("color", colors_rgba.flatten())


def create_vertex_color_material(obj, material_name="VertexColorMaterial"):
    """
    Create a material that renders vertex colors through per-channel Map Range nodes.

    Parameters
    ----------
    obj : bpy.types.Object
        Object with at least one vertex color layer.
    material_name : str, optional
        Name for the new material.
    """
    if not obj.data.color_attributes:
        raise ValueError("The object has no vertex color layers.")

    material = bpy.data.materials.new(name=material_name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    for node in nodes:
        nodes.remove(node)

    vertex_color_node = nodes.new(type="ShaderNodeVertexColor")
    vertex_color_node.layer_name = obj.data.color_attributes[0].name
    vertex_color_node.location = (-1000, 0)

    separate_color_node = nodes.new(type="ShaderNodeSeparateColor")
    separate_color_node.location = (-800, 0)

    map_range_r = nodes.new(type="ShaderNodeMapRange")
    map_range_r.label = "Map Range R"
    map_range_r.location = (-600, 300)

    map_range_g = nodes.new(type="ShaderNodeMapRange")
    map_range_g.label = "Map Range G"
    map_range_g.location = (-600, 0)

    map_range_b = nodes.new(type="ShaderNodeMapRange")
    map_range_b.label = "Map Range B"
    map_range_b.location = (-600, -300)

    combine_rgb = nodes.new(type="ShaderNodeCombineColor")
    combine_rgb.location = (-200, 0)

    bsdf_node = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf_node.location = (0, 0)

    output_node = nodes.new(type="ShaderNodeOutputMaterial")
    output_node.location = (400, 0)

    links.new(vertex_color_node.outputs["Color"], separate_color_node.inputs["Color"])
    links.new(separate_color_node.outputs["Red"], map_range_r.inputs["Value"])
    links.new(separate_color_node.outputs["Green"], map_range_g.inputs["Value"])
    links.new(separate_color_node.outputs["Blue"], map_range_b.inputs["Value"])
    links.new(map_range_r.outputs["Result"], combine_rgb.inputs["Red"])
    links.new(map_range_g.outputs["Result"], combine_rgb.inputs["Green"])
    links.new(map_range_b.outputs["Result"], combine_rgb.inputs["Blue"])
    links.new(combine_rgb.outputs["Color"], bsdf_node.inputs["Base Color"])
    links.new(bsdf_node.outputs["BSDF"], output_node.inputs["Surface"])

    for map_range_node in [map_range_r, map_range_g, map_range_b]:
        map_range_node.inputs["From Min"].default_value = 0.0
        map_range_node.inputs["From Max"].default_value = 1.0
        map_range_node.inputs["To Min"].default_value = 0.0
        map_range_node.inputs["To Max"].default_value = 1.0

    obj.active_material = material
