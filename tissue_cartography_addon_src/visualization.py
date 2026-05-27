"""
Visualization utilities: bounding boxes, ortho-slice planes, materials, and vertex shading.
"""

import bpy
import numpy as np

from .io_utils import normalize_quantiles


# ---------------------------------------------------------------------------
# Create Box to represent 3D image data bounds
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Slice planes for image visualization
# ---------------------------------------------------------------------------


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
    plane.name = f"Slice_{axis.upper()}_{position:.1f}"
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


# ---------------------------------------------------------------------------
# Mesh–plane intersection (analytic edge-plane intersection)
# ---------------------------------------------------------------------------

# Maps intersection curve object name → (surface_mesh_name, slice_plane_name).
# Used by the depsgraph handler to know which objects to track and update.
_intersection_trackers: dict = {}
_is_updating = False  # re-entrancy guard for the depsgraph handler


def compute_plane_intersection_segments(surface_mesh, plane_obj, z_offset=0.0):
    """
    Compute line segments where the surface mesh intersects an axis-aligned plane object.

    Derives the cutting-plane equation from the world-space positions of ``plane_obj``'s
    vertices, then tests every triangulated edge of ``surface_mesh`` against the plane.
    For each triangle with two crossing edges the two interpolated intersection points form
    one segment. Fully vectorised with numpy; works on non-watertight and flipped-normal
    meshes, and handles slice planes whose local origin is not at the cutting position
    (as produced by :func:`create_slice_plane`).

    Parameters
    ----------
    surface_mesh : bpy.types.Object
        Mesh whose intersection with the plane is computed.
    plane_obj : bpy.types.Object
        A planar mesh object defining the cutting plane (e.g. created by
        :func:`create_slice_plane`). Its current ``matrix_world`` is used, so
        the function is correct even after the object has been moved.
    z_offset : float, optional
        Small offset applied along the plane's world-space normal to all intersection
        points before returning them. Lifts the curve slightly above the slice plane
        surface to avoid z-fighting in the viewport.

    Returns
    -------
    np.ndarray of shape (K, 2, 3)
        World-space endpoints of K intersection line segments.
        Empty array of shape (0, 2, 3) when there is no intersection.
    """
    # --- Plane equation from plane_obj's world-space vertex positions ---
    # Using world-space vertices directly is correct regardless of how the local
    # vertex positions and matrix_world were set (e.g. after transform_apply).
    p0 = np.array(plane_obj.matrix_world @ plane_obj.data.vertices[0].co)
    p1 = np.array(plane_obj.matrix_world @ plane_obj.data.vertices[1].co)
    p2 = np.array(plane_obj.matrix_world @ plane_obj.data.vertices[2].co)
    normal = np.cross(p1 - p0, p2 - p0)
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-12:
        return np.zeros((0, 2, 3), dtype=np.float64)
    normal /= norm_len  # unit normal in world space

    # --- Surface mesh vertices in world space ---
    mw_a = np.array(surface_mesh.matrix_world)
    n_verts = len(surface_mesh.data.vertices)
    verts_local_a = np.empty(n_verts * 3, dtype=np.float32)
    surface_mesh.data.vertices.foreach_get("co", verts_local_a)
    verts_local_a = verts_local_a.reshape(n_verts, 3)
    verts_world = (mw_a[:3, :3] @ verts_local_a.T).T + mw_a[:3, 3]  # (N, 3)

    # --- Signed distances from the plane ---
    d = (verts_world - p0) @ normal  # (N,)

    # --- Triangulated faces ---
    surface_mesh.data.calc_loop_triangles()
    n_tris = len(surface_mesh.data.loop_triangles)
    if n_tris == 0:
        return np.zeros((0, 2, 3), dtype=np.float64)
    tri_verts_flat = np.empty(n_tris * 3, dtype=np.int32)
    surface_mesh.data.loop_triangles.foreach_get("vertices", tri_verts_flat)
    tris = tri_verts_flat.reshape(n_tris, 3)  # (M, 3)

    # --- Vectorised edge-plane crossing ---
    # Three directed edges per triangle: 0→1, 1→2, 2→0
    a_cols = np.array([0, 1, 2])
    b_cols = np.array([1, 2, 0])
    va_idx = tris[:, a_cols]  # (M, 3) vertex indices for edge starts
    vb_idx = tris[:, b_cols]  # (M, 3) vertex indices for edge ends
    da = d[va_idx]            # (M, 3) signed distances at edge starts
    db = d[vb_idx]            # (M, 3) signed distances at edge ends
    crosses = da * db < 0     # (M, 3) True when edge straddles the plane

    n_crosses = crosses.sum(axis=1)  # (M,)
    valid = n_crosses == 2           # one segment per intersecting triangle

    if not valid.any():
        return np.zeros((0, 2, 3), dtype=np.float64)

    denom = da - db
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)  # avoid div-by-zero
    t = da / denom  # (M, 3) interpolation parameter along each edge

    va_pos = verts_world[va_idx]  # (M, 3, 3) world-space edge starts
    vb_pos = verts_world[vb_idx]  # (M, 3, 3) world-space edge ends
    pts = va_pos + t[:, :, np.newaxis] * (vb_pos - va_pos)  # (M, 3, 3) world space

    # Extract the 2 crossing-edge intersection points per valid triangle.
    # argsort(~crosses) sorts False(=0) before True(=1), so the FIRST 2 indices
    # of the sorted result are the positions of the 2 True (crossing) entries.
    valid_pts = pts[valid]        # (K, 3, 3)
    valid_cross = crosses[valid]  # (K, 3)
    cross_idx = np.argsort(~valid_cross, axis=1)[:, :2]  # (K, 2)
    K = int(valid.sum())
    row_idx = np.arange(K)[:, np.newaxis]
    segments = valid_pts[row_idx, cross_idx]  # (K, 2, 3) world-space endpoints

    # Lift slightly above the slice plane to avoid z-fighting
    if z_offset != 0.0:
        segments = segments + z_offset * normal  # broadcast (K, 2, 3) + (3,)

    return segments


def _make_red_emission_material(name):
    """Return a new red emission material suitable for intersection visualizations."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for node in list(nodes):
        if node.type != 'OUTPUT_MATERIAL':
            nodes.remove(node)
    output = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
    if output is None:
        output = nodes.new('ShaderNodeOutputMaterial')
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Color'].default_value = (1.0, 0.0, 0.0, 1.0)
    emission.inputs['Strength'].default_value = 3.0
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    return mat


def _build_curve_splines(curve_data, segments):
    """
    Rebuild all POLY splines of *curve_data* from a (K, 2, 3) segments array.

    Clears existing splines first. Safe to call with an empty segments array.
    """
    curve_data.splines.clear()
    for seg in segments:
        sp = curve_data.splines.new('POLY')
        sp.points.add(1)  # POLY spline starts with 1 point; add a second
        sp.points[0].co = (*seg[0], 1.0)
        sp.points[1].co = (*seg[1], 1.0)


def _plane_and_ancestors(plane_name):
    """Yield plane_name and the names of all its ancestors (for parent-move detection)."""
    obj = bpy.data.objects.get(plane_name)
    while obj:
        yield obj.name
        obj = obj.parent


def _update_intersection_handler(scene, depsgraph):
    """
    depsgraph_update_post handler: recompute and redisplay all registered
    mesh–plane intersection curves when either tracked object changes.

    Also fires when a *parent* of the slice plane is moved (e.g. the bounding
    box is translated together with all its child slice planes).
    """
    global _is_updating
    if _is_updating or not depsgraph.id_type_updated('OBJECT'):
        return

    updated_names = {
        upd.id.name
        for upd in depsgraph.updates
        if isinstance(upd.id, bpy.types.Object)
    }

    _is_updating = True
    try:
        to_remove = []
        for obj_name, (mesh_name, plane_name) in list(_intersection_trackers.items()):
            # Clean up stale entries whose objects have been deleted
            if (obj_name not in bpy.data.objects
                    or mesh_name not in bpy.data.objects
                    or plane_name not in bpy.data.objects):
                to_remove.append(obj_name)
                continue
            # Recompute if the surface mesh changed, the plane changed directly,
            # or any ancestor of the plane changed (covers "move bounding box" case).
            plane_affected = not updated_names.isdisjoint(_plane_and_ancestors(plane_name))
            if mesh_name not in updated_names and not plane_affected:
                continue
            curve_obj = bpy.data.objects[obj_name]
            z_offset = float(curve_obj.get("_tc_z_offset", 0.0))
            surface_mesh = bpy.data.objects[mesh_name]
            plane_obj = bpy.data.objects[plane_name]
            segments = compute_plane_intersection_segments(
                surface_mesh, plane_obj, z_offset=z_offset
            )
            _build_curve_splines(curve_obj.data, segments)

        for name in to_remove:
            del _intersection_trackers[name]
    finally:
        _is_updating = False

    if not _intersection_trackers:
        bpy.app.handlers.depsgraph_update_post.remove(_update_intersection_handler)


def create_intersection_line_visualization(slice_plane, surface_mesh):
    """
    Create a live intersection visualization between a slice plane and a surface mesh.

    Computes the intersection analytically by testing each triangulated mesh edge
    against the cutting plane (see :func:`compute_plane_intersection_segments`).
    The result is displayed as a CURVE object with a bevel giving it finite thickness
    and a red emission material. A ``depsgraph_update_post`` handler keeps the curve
    updated whenever either object is moved or edited.

    Unlike the Boolean modifier approach this works correctly on non-watertight meshes
    and meshes with flipped normals.

    Parameters
    ----------
    slice_plane : bpy.types.Object
        The slice plane object (already placed in the scene).
    surface_mesh : bpy.types.Object
        The surface mesh to intersect with.

    Returns
    -------
    bpy.types.Object
        The intersection curve object, named ``Intersect_{slice_plane.name}``.
    """
    # line thickness is heuristically chosen based on overall scene scale
    thickness = float(np.mean(compute_edge_lengths(slice_plane)) / 500) 
    z_offset = 0 # thickness # move curve slightly above plane
    segments = compute_plane_intersection_segments(surface_mesh, slice_plane,
                                                   z_offset=z_offset)

    # Create the CURVE object
    curve_name = f"Intersect_{slice_plane.name}"
    curve_data = bpy.data.curves.new(curve_name, type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.bevel_depth = thickness
    curve_data.bevel_resolution = 2
    curve_data.use_fill_caps = True

    _build_curve_splines(curve_data, segments)

    obj = bpy.data.objects.new(curve_name, curve_data)
    bpy.context.collection.objects.link(obj)
    obj.show_in_front = True

    # Red emission material
    mat = _make_red_emission_material(f"IntersectionEmission_{slice_plane.name}")
    obj.data.materials.append(mat)

    # Store tracking info as custom properties so the handler can read them
    obj["_tc_surface_mesh"] = surface_mesh.name
    obj["_tc_slice_plane"] = slice_plane.name
    obj["_tc_z_offset"] = z_offset

    # Register live-update handler and record this object in the tracker
    _intersection_trackers[obj.name] = (surface_mesh.name, slice_plane.name)
    if _update_intersection_handler not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_update_intersection_handler)

    return obj


# ---------------------------------------------------------------------------
# Projecting image data onto mesh vertices
# ---------------------------------------------------------------------------


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
    The new attribute obj.data.color_attributes["VertexColor"].

    Parameters
    ----------
    obj : bpy.types.Object
        Mesh object.
    colors : array-like of shape (n_vertices, 3)
        RGB values in [0, 1] for each vertex.

    Returns
    -------
    None
    """
    if obj.type != 'MESH':
        raise ValueError("Object is not a mesh.")
    ca = obj.data.color_attributes
    if "VertexColor" not in {a.name for a in ca}:
        ca.new(name="VertexColor", type='FLOAT_COLOR', domain='CORNER')
    n = len(obj.data.loops)
    loop_vert_flat = np.zeros(n, dtype=np.int32)
    obj.data.loops.foreach_get("vertex_index", loop_vert_flat)
    colors_rgba = np.ones((n, 4), dtype=np.float32)
    colors_rgba[:, :3] = np.asarray(colors, dtype=np.float32)[loop_vert_flat]
    ca["VertexColor"].data.foreach_set("color", colors_rgba.flatten())


def create_vertex_color_material(obj, material_name="VertMat"):
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
    vertex_color_node.layer_name = "VertexColor"
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


# ---------------------------------------------------------------------------
# Projecting image data onto mesh using UV maps
# ---------------------------------------------------------------------------


def create_material_from_multilayer_array(mesh, array, material_name="ProjMat"):
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

    # Create material first so we can use its actual Blender-assigned name (which may be
    # auto-incremented, e.g. "ProjectedMaterial_MeshA.001") as a unique prefix for image
    # names. Without this, images from different projections all share the same generic
    # names ("Channel_0_Layer_0", …), and Blender may re-generate (clearing to black) any
    # GENERATED image when a new image with the same base name is created.
    material = bpy.data.materials.new(name=material_name)
    actual_name = material.name  # may differ from material_name if a duplicate existed

    images = {}
    for ic, channel in enumerate(array_normalized):
        for il, layer in enumerate(channel):
            pixel_data = np.zeros((image_height, image_width, 4), dtype=np.float32)
            pixel_data[..., 0] = pixel_data[..., 1] = pixel_data[..., 2] = layer[::-1]
            pixel_data[..., 3] = 1.0
            img = bpy.data.images.new(
                name=f"{actual_name}_Channel_{ic}_Layer_{il}",
                width=image_width, height=image_height,
            )
            img.pixels.foreach_set(pixel_data.flatten())
            # Pack the image so its source changes from GENERATED to PACKED. This embeds
            # the pixel data in the .blend file and prevents Blender from regenerating
            # the image (which would produce a black result) on subsequent operations.
            img.pack()
            images[(ic, il)] = img
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