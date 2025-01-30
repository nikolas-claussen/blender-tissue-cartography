bl_info = {
    "name": "Tissue Cartography",
    "blender": (4, 3, 0),
    "category": "Scene",
}

import bpy
from bpy.props import StringProperty, FloatVectorProperty, IntVectorProperty, FloatProperty, IntProperty, BoolProperty, EnumProperty
from bpy.types import Operator, Panel
import mathutils
import bmesh

from pathlib import Path
import os
import numpy as np
import difflib
import itertools
import subprocess
import sys

import tifffile
from scipy import interpolate, ndimage, spatial, stats, linalg
from skimage import measure


### Installing dependencies

def install_dependencies():
    try:
        import scipy
        import skimage
    except ImportError:
        python_executable = sys.executable
        subprocess.check_call([python_executable, "-m", "pip", "install", "scipy", "scikit-image", "tifffile"])


### I/O and image handling


def load_png(image_path):
    """Load .png into numpy array."""
    image = bpy.data.images.load(image_path)
    width, height = image.size
    pixels = np.array(image.pixels[:], dtype=np.float32)
    return pixels.reshape((height, width, -1))


def normalize_quantiles(image, quantiles=(0.01, 0.99), channel_axis=None, clip=False,
                        data_type=None):
    """
    Normalize a multi-dimensional image by setting given quantiles to 0 and 1.
    
    Parameters
    ----------
    image : np.array
        Multi-dimensional image.
    quantiles : tuple
        Image quantile to set to 0 and 1.
    channel_axis : int or None
        If None, the image is assumed to have only a single channel.
        If int, indicates the position of the channel axis. 
        Each channel is normalized separately.
    clip : bool
        Whether to clip image to 0-1. Automatically enabled if converting to int dtype.
    data_type : None, np.unit8 or np.uint16
        If not None, image is converted to give data type.
    
    Returns
    -------
    image_normalized : np.array
        Normalized image, the same shape as input
    """
    if channel_axis is None:
        image_normalized = image - np.nanquantile(image, quantiles[0])
        image_normalized /= np.nanquantile(image_normalized, quantiles[1])
        image_normalized = np.nan_to_num(image_normalized)
    else:
        image_normalized = np.moveaxis(image, channel_axis, 0)
        image_normalized = np.stack([ch - np.nanquantile(ch, quantiles[0]) for ch in image_normalized])
        image_normalized = np.stack([ch / np.nanquantile(ch, quantiles[1]) for ch in image_normalized])
        image_normalized = np.moveaxis(np.nan_to_num(image_normalized), 0, channel_axis)
    if clip or (data_type is not None):
        image_normalized = np.clip(image_normalized, 0, 1)
    if data_type is np.uint8:
        image_normalized = np.round((2**8-1)*image_normalized).astype(np.uint8)
    if data_type is np.uint16:
        image_normalized = np.round((2**16-1)*image_normalized).astype(np.uint16)
    return image_normalized


### Tissue cartography - projecting 3d images to UV textures


def get_uv_layout(obj, uv_layout_path, image_resolution):
    """Get UV layout mask for obj object as a np.array. As a side effect, saves layout to disk and deselects everything except obj."""
    if os.path.exists(uv_layout_path):
        os.remove(uv_layout_path)

    bpy.ops.object.select_all(action='DESELECT')  # Deselect all objects
    obj.select_set(True)  # Select the specific object
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    
    # Set all faces to selected for the UV layout
    mesh = bmesh.from_edit_mesh(obj.data)
    for face in mesh.faces:
        face.select = True
    bmesh.update_edit_mesh(obj.data)
    
    bpy.ops.uv.export_layout(filepath=uv_layout_path, size=(image_resolution, image_resolution), opacity=1, export_all=False, check_existing=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    UV_layout = load_png(uv_layout_path)
    
    return (UV_layout.sum(axis=-1) > 0)[::-1]


def get_uv_normal_world_per_loop(mesh_obj, filter_unique=False):
    """
    Get UV, normals, and world and normal for each loop (half-edge) as np.array.
    
    If filter_unique, remove "duplicate" loops (for which UV, normals and position
    are identical).
    """
    if not mesh_obj:
        raise TypeError("No object selected")
    if mesh_obj.type != 'MESH':
        raise TypeError("Selected object is not a mesh")
    world_matrix = mesh_obj.matrix_world
    uv_layer = mesh_obj.data.uv_layers.active
    if not uv_layer:
        raise RuntimeError("Mesh does not have an active UV map")
    loop_uvs = np.zeros((len(mesh_obj.data.loops), 2), dtype=np.float32)
    loop_normals = np.zeros((len(mesh_obj.data.loops), 3), dtype=np.float32)
    loop_world_positions = np.zeros((len(mesh_obj.data.loops), 3), dtype=np.float32)
    for loop in mesh_obj.data.loops:
        loop_uvs[loop.index] = uv_layer.data[loop.index].uv
        loop_normals[loop.index] = world_matrix.to_3x3() @ mesh_obj.data.vertices[loop.vertex_index].normal
        loop_world_positions[loop.index] = world_matrix @ mesh_obj.data.vertices[loop.vertex_index].co
    if filter_unique:
        unqiue_loops = np.unique(np.hstack([loop_uvs, loop_normals, loop_world_positions]), axis=0)
        loop_uvs, loop_normals, loop_world_positions = (unqiue_loops[:,:2], unqiue_loops[:,2:5], unqiue_loops[:,5:])
    loop_normals = np.round((loop_normals.T/np.linalg.norm(loop_normals, axis=1)).T, decimals=4)
    return loop_uvs, loop_normals, loop_world_positions


def bake_per_loop_values_to_uv(loop_uvs, loop_values, image_resolution):
    """
    Bake (interpolate) values (normals or world position) defined per loop into the UV square.
    
    UV coordinates outside [0,1] are ignored.
    
    Parameters
    ----------
    loop_uvs : np.array of shape (n_loops, 2)
        UV coordinates of loop.
    loop_values : np.array of shape (n_loops, ...)
        Input field. Can be an array with any number of axes (e.g. scalar or vector field).
    image_resolution : int, default 256
        Size of UV grid. Determines resolution of result.

    Returns
    -------
    
    interpolated : np.array of shape (uv_grid_steps, uv_grid_steps, ...)
        Field across [0,1]**2 UV grid, with a uniform step size. UV positions that don't
        correspond to any value are set to np.nan.
            
    """
    U, V = np.meshgrid(*(2*(np.linspace(0,1, image_resolution),)))
    interpolated = interpolate.griddata(loop_uvs, loop_values, (U, V), method='linear')[::-1]
    return interpolated


def bake_volumetric_data_to_uv(image, baked_world_positions, resolution, baked_normals, normal_offsets=(0,), affine_matrix=None):
    """ 
    Interpolate volumetric image data onto UV coordinate grid.
    
    Uses baked 3d world positions corresponding to each UV grid point (see bake_per_loop_values_to_UV).
    3d coordinates (in microns) are converted into image coordinates via the resolution scaling factor.
    The resolution of the bake (number of pixels) is determined by the shape of baked_world_positions.
    
    normal_offsets moves the 3d positions whose volumetric voxel values will be baked inwards or outwards
    along the surface normal. Providing a list of offsets results in a multi-layer pullback
    
    Parameters
    ----------
    image : 4d np.array
        Image, axis 0  is assumed to be the channel axis
    baked_world_positions : np.array of shape (image_resolution, image_resolution, uv_grid_steps, 3)
        3d world positions baked to UV grid, with uniform step size. UV positions that don't correspond to 
        any value are set to np.nan.
    resolution : np.array of shape (3,)
        Resolution in pixels/microns for each of the three spatial axes.
    baked_normals : np.array of shape (image_resolution, image_resolution, uv_grid_steps, 3)
        3d world normals baked to UV grid, with uniform step size. UV positions that don't correspond to 
        any value are set to np.nan.
    normal_offsets : np.array of shape (n_layers,), default (0,)
        Offsets along normal direction, in same units as interpolated_3d_positions (i.e. microns).
        0 corresponds to no shift.
    affine_matrix : np.array of shape (4, 4) or None
        If not None, transform coordinates by affine trafor before calling interpolator
        
    Returns
    -------
    aked_data : np.array of shape (n_channels, n_layers, image_resolution, image_resolution)
        Multi-layer 3d volumetric data baked onto UV.
    """
    x, y, z = [np.arange(ni) for ni in image.shape[1:]]
    baked_data = []
    for o in normal_offsets:
        position = (baked_world_positions+o*baked_normals)
        if affine_matrix is not None:
            position = position @ affine_matrix[:3, :3].T + affine_matrix[:3,3]
        position =  position/resolution
        baked_layer_data = np.stack([interpolate.interpn((x, y, z), channel, position,
                                     method="linear", bounds_error=False) for channel in image])
        baked_data.append(baked_layer_data)
    baked_data = np.stack(baked_data, axis=1)
    return baked_data


### Bounding box and orthoslices for visualizing the 3d data


def create_box(length, width, height, name="RectangularBox", hide=True):
    """
    Creates a rectangular box using Blender's default cube.
    One corner is positioned at the origin, and the box lies in the positive x/y/z quadrant.

    Args:
        length (float): Length of the box along the X-axis.
        width (float): Width of the box along the Y-axis.
        height (float): Height of the box along the Z-axis.
    """
    # Store the current active object
    current_active = bpy.context.active_object

    bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = (length / 2, width / 2, height / 2)
    obj.location = (length / 2, width / 2, height / 2)
    bpy.ops.object.transform_apply(location=True, scale=True)
    obj.hide_set(hide)
    # re-select the currently active object
    if current_active:
        bpy.context.view_layer.objects.active = current_active
    return obj


def create_slice_plane(length, width, height, axis='z', position=0.0):
    """
    Creates a 2D plane as a slice of a rectangular box along a specified axis.
    The plane lies within the bounds of the box.

    Args:
        length (float): Length of the box along the X-axis.
        width (float): Width of the box along the Y-axis.
        height (float): Height of the box along the Z-axis.
        axis (str): Axis along which to slice ('x', 'y', or 'z').
        position (float): Position along the chosen axis for the slice plane.
                          Should be within the range of the box dimensions.
    """
    current_active = bpy.context.active_object
    # Validate axis and position
    if axis not in {'x', 'y', 'z'}:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")
    
    axis_limits = {'x': length, 'y': width, 'z': height}
    if not (0.0 <= position <= axis_limits[axis]):
        raise ValueError(f"Position must be within [0, {axis_limits[axis]}] for axis {axis}.")

    # Create the plane's dimensions based on the slicing axis
    if axis == 'x':
        plane_size = (height, width) #(width, height)
        location =  (position, width / 2, height / 2)
        rotation = (0, 1.5708, 0)  # Rotate to align with the YZ-plane
    elif axis == 'y':
        plane_size = (length, height)
        location = (length / 2, position, height / 2)
        rotation = (1.5708, 0, 0)  # Rotate to align with the XZ-plane
    else:  # 'z'
        plane_size = (length, width)
        location = (length / 2, width / 2, position)
        rotation = (0, 0, 0)  # No rotation needed for the XY-plane

    # Add a plane
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 0))
    plane = bpy.context.active_object
    plane.name = f"SlicePlane_{axis.upper()}_{position:.2f}"

    # Scale and position the plane
    plane.scale = (plane_size[0] / 2, plane_size[1] / 2, 1)
    plane.location = location
    plane.rotation_euler = rotation

    # Apply transformations (scale, location, rotation)
    bpy.ops.object.transform_apply(location=True, scale=True, rotation=True)

    # Restore the previously active object
    if current_active:
        bpy.context.view_layer.objects.active = current_active

    return plane


def get_slice_image(image_3d, resolution, axis='z', position=0.0):
    """Get slice of 3d image along axis for ortho-slice visualization.
    image_3d must be a 4d array (channels, x, y, z). Position in microns."""
    if axis == 'x':
        ind = int(np.round(position / resolution[0]))
        slice_img = image_3d[:,ind,:,::-1]
    elif axis == 'y':
        ind = int(np.round(position / resolution[1]))
        slice_img = image_3d[:,:,ind,:].transpose((0,2,1))
    elif axis == 'z': 
        ind = int(np.round(position / resolution[0]))
        slice_img = image_3d[:,:,:,ind].transpose((0,2,1))
    return slice_img


def create_material_from_array(slice_plane, array, material_name="SliceMaterial"):
    """
    Creates a material for a ortho-slice plane using a 2D numpy array as a texture.

    Args:
        slice_plane (bpy.types.Object): The plane object to which the material will be applied.
        array (numpy.ndarray): 2D array representing grayscale values (0-1), or 3D array representing RGBA values (0-1).
        material_name (str): Name of the new material.
    """
    # Validate input array
    if not len(array.shape) in [2,3]:
        raise ValueError("Input array must be 2D.")
    
    # Normalize array to range [0, 1] and convert to a flat list
    image_height, image_width = array.shape[:2]
    pixel_data = np.zeros((image_height, image_width, 4), dtype=np.float32)  # RGBA
    if len(array.shape) == 2:
        pixel_data[..., 0] =  pixel_data[..., 1] = pixel_data[..., 2] = array
        pixel_data[..., 3] = 1.0  # Alpha
    else:
        pixel_data[...] = array
    pixel_data = pixel_data.flatten()

    # Create a new image in Blender
    image = bpy.data.images.new(name="SliceTexture", width=image_width, height=image_height)
    image.pixels = pixel_data.tolist()

    # Create a new material
    material = bpy.data.materials.new(name=material_name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    # Clear default nodesx
    for node in nodes:
        nodes.remove(node)

    # Add required nodes
    texture_node = nodes.new(type="ShaderNodeTexImage")
    texture_node.image = image
    bsdf_node = nodes.new(type="ShaderNodeBsdfPrincipled")
    output_node = nodes.new(type="ShaderNodeOutputMaterial")

    # Arrange nodes
    texture_node.location = (-400, 0)
    bsdf_node.location = (0, 0)
    output_node.location = (400, 0)

    # Connect nodes
    links.new(texture_node.outputs["Color"], bsdf_node.inputs["Base Color"])
    links.new(bsdf_node.outputs["BSDF"], output_node.inputs["Surface"])

    # Assign the material to the plane
    slice_plane.active_material = material
    return None


### Pullback shading


def create_material_from_multilayer_array(mesh, array, material_name="ProjectedMaterial"):
    """
    Creates a material for a mesh using multi-channel, multi-layer projection.

    Args:
        obj (bpy.types.Object): The mesh object to which the material will be applied.
        array (numpy.ndarray): 4D array of shape (channels, layers, U, V)
        material_name (str): Name of the new material.
    """
    # Validate and normalize input array
    if not len(array.shape) == 4:
        raise ValueError("Input array must have 4 axes.")
    array_normalized = normalize_quantiles(array, quantiles=(0.01, 0.99), channel_axis=0,
                                           clip=True, data_type=None)
    # Create a new image in Blender for each layer and channel
    image_height, image_width = array.shape[-2:]
    n_channels, n_layers = array.shape[:2]
    images = {}
    for ic, chanel in enumerate(array_normalized):
        for il, layer in enumerate(chanel):
            pixel_data = np.zeros((image_height, image_width, 4), dtype=np.float32)
            pixel_data[..., 0] =  pixel_data[..., 1] = pixel_data[..., 2] = layer[::-1]
            pixel_data[..., 3] = 1.0  # Alpha
            pixel_data = pixel_data.flatten()
            images[(ic, il)] = bpy.data.images.new(name=f"Channel_{ic}_Layer_{il}",
                                                   width=image_width, height=image_height)
            images[(ic, il)].pixels = pixel_data.tolist()
    # Create a new material
    material = bpy.data.materials.new(name=material_name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    # Clear default nodesx
    for node in nodes:
        nodes.remove(node)
    # Add required nodes
    texture_nodes = {}
    for (ic, il), image in images.items():
        texture_nodes[(ic, il)] = nodes.new(type="ShaderNodeTexImage")
        texture_nodes[(ic, il)].image = image
        texture_nodes[(ic, il)].location = (-400, ic*400 + il*300)
    
    bsdf_node = nodes.new(type="ShaderNodeBsdfPrincipled")
    output_node = nodes.new(type="ShaderNodeOutputMaterial")

    # Arrange nodes
    bsdf_node.location = (0, 0)
    output_node.location = (400, 0)

    # Connect nodes
    links.new(texture_nodes[(0,0)].outputs["Color"], bsdf_node.inputs["Base Color"])
    links.new(bsdf_node.outputs["BSDF"], output_node.inputs["Surface"])

    # Assign the material to the mesh
    mesh.active_material = material
    return None


### Vertex shading


def compute_edge_lengths(obj):
    """
    Computes the lengths of all edges in a mesh object as a numpy array.

    Args:
        obj (bpy.types.Object): The mesh object to compute edge lengths for.

    Returns:
        numpy.ndarray: A 1D array containing the lengths of all edges in the mesh.
    """
    # Ensure the object is a mesh
    if obj.type != 'MESH':
        raise ValueError("The selected object is not a mesh.")
    # Ensure the mesh is in edit mode for accurate vertex data
    bpy.context.view_layer.objects.active = obj
    if obj.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    edge_lengths = []
    for edge in obj.data.edges:
        v1 = obj.data.vertices[edge.vertices[0]].co
        v2 = obj.data.vertices[edge.vertices[1]].co
        edge_lengths.append((v1 - v2).length)
    return np.array(edge_lengths)


def get_image_to_vertex_interpolator(obj, image_3d, resolution_array, quantiles=(0.01, 0.99)):
    """
    Get interpolator that maps vertex position -> image intensity.
    
    Returns a list of interpolators, one for each channel.
    To avoid aliasing, the 3d image is smoothed with
    sigma=median edge length /2. The image data is also normalized to
    range from 0-1 using the provided quantiles.
    """
    anti_aliasing_scale = np.median(compute_edge_lengths(obj))/2
    image_3d_smoothed = np.stack([ndimage.gaussian_filter(ch, anti_aliasing_scale/resolution_array)
                                  for ch in image_3d])
    image_3d_smoothed = normalize_quantiles(image_3d_smoothed,
                                            quantiles=quantiles, clip=True, data_type=None)
    x, y, z = [np.arange(ni)*resolution_array[i]
               for i, ni in enumerate(image_3d.shape[1:])]
    
    return [interpolate.RegularGridInterpolator((x,y,z), ch, method='linear', bounds_error=False)
            for ch in image_3d_smoothed]


def assign_vertex_colors(obj, colors):
    """
    Assigns an RGB color to each vertex in the given object.
    Args:
        obj: The mesh object.
        colors: A list or dict of (R, G, B) tuples for each vertex.
    """
    if obj.type != 'MESH':
        print("Object is not a mesh!")
        return
    if not obj.data.vertex_colors:
        obj.data.vertex_colors.new()
    color_layer = obj.data.vertex_colors.active
    # Assign colors to each loop (face corner)
    for loop in obj.data.loops:    
        color_layer.data[loop.index].color = (*colors[loop.vertex_index], 1.0)  # RGBA
    return None


def create_vertex_color_material(object, material_name="VertexColorMaterial"):
    """
    Creates a material for an object that uses vertex colors.
    The R, G, and B channels are processed through separate "Map Range" nodes
    to edit their brightness, and then combined into a Principled BSDF.

    Args:
        object (bpy.types.Object): The object to which the material will be applied.
        material_name (str): Name of the new material.
    """
    # Ensure the object has a vertex color layer
    if not object.data.vertex_colors:
        raise ValueError("The object has no vertex color layers.")

    # Create a new material
    material = bpy.data.materials.new(name=material_name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    # Clear default nodes
    for node in nodes:
        nodes.remove(node)

    # Add nodes
    vertex_color_node = nodes.new(type="ShaderNodeVertexColor")
    vertex_color_node.layer_name = object.data.vertex_colors[0].name
    vertex_color_node.location = (-1000, 0)

    separate_color_node = nodes.new(type="ShaderNodeSeparateRGB")
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

    combine_rgb = nodes.new(type="ShaderNodeCombineRGB")
    combine_rgb.location = (-200, 0)

    bsdf_node = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf_node.location = (000, 0)

    output_node = nodes.new(type="ShaderNodeOutputMaterial")
    output_node.location = (400, 0)

    # Connect nodes
    links.new(vertex_color_node.outputs["Color"], separate_color_node.inputs["Image"])
    links.new(separate_color_node.outputs["R"], map_range_r.inputs["Value"])
    links.new(separate_color_node.outputs["G"], map_range_g.inputs["Value"])
    links.new(separate_color_node.outputs["B"], map_range_b.inputs["Value"])

    links.new(map_range_r.outputs["Result"], combine_rgb.inputs["R"])
    links.new(map_range_g.outputs["Result"], combine_rgb.inputs["G"])
    links.new(map_range_b.outputs["Result"], combine_rgb.inputs["B"])

    links.new(combine_rgb.outputs["Image"], bsdf_node.inputs["Base Color"])
    links.new(bsdf_node.outputs["BSDF"], output_node.inputs["Surface"])

    # Set default map range values for each channel
    for map_range_node in [map_range_r, map_range_g, map_range_b]:
        map_range_node.inputs["From Min"].default_value = 0.0
        map_range_node.inputs["From Max"].default_value = 1.0
        map_range_node.inputs["To Min"].default_value = 0.0
        map_range_node.inputs["To Max"].default_value = 1.0

    # Assign the material to the object
    object.active_material = material
    return None


### Marching cubes


def create_mesh_from_numpy(name, verts, faces):
    """
    Creates a Blender mesh object from NumPy arrays of vertices and faces.
    
    :param name: Name of the new mesh object.
    :param verts: NumPy array of shape (n, 3) containing vertex coordinates.
    :param faces: NumPy array of shape (m, 3 or 4) containing face indices.
    :return: The created mesh object.
    """
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    # Link the object to the scene
    bpy.context.collection.objects.link(obj)
    mesh.from_pydata(verts.tolist(), [], faces.tolist())
    mesh.update()
    return obj


### Iterative closest point alignment


def package_affine_transformation(matrix, vector):
    """Package matrix transformation & translation into (d+1,d+1) matrix representation of affine transformation."""
    matrix_rep = np.hstack([matrix, vector[:, np.newaxis]])
    matrix_rep = np.pad(matrix_rep, ((0,1),(0,0)), constant_values=0)
    matrix_rep[-1,-1] = 1
    return matrix_rep


def get_inertia(pts):
    """Get inertia tensor of 3d point cloud."""
    pts_nomean = pts - np.mean(pts, axis=0)
    x, y, z = pts_nomean.T
    Ixx = np.mean(x**2)
    Ixy = np.mean(x*y)
    Ixz = np.mean(x*z)
    Iyy = np.mean(y**2)
    Iyz = np.mean(y*z)
    Izz = np.mean(z*z)
    return np.array([[Ixx, Ixy, Ixz], [Ixy,Iyy, Iyz], [Ixz, Iyz, Izz]])


def align_by_centroid_and_intertia(source, target, scale=True, shear=True, improper=False, n_samples=10000):
    """
    Align source point cloud to target point cloud using affine transformation.
    
    Align by matching centroids and axes of inertia tensor. Since the inertia tensor is invariant
    under reflections along its principal axes, all 2^3 reflections are tried and the one leading
    to the best agreement with the target is chosen.
    
    Parameters
    ----------
    source : np.array of shape (n_source, 3)
        Point cloud to be aligned.
    target : np.array of shape (n_target, 3)
        Point cloud to align to.
    scale : bool, default True
        Whether to allow scale transformation (True) or rotations only (False)
    shear : bool, default False
        Whether to allow shear transformation (True) or rotations/scale only (False)
    improper : bool, default False
        Whether to allow transfomations with determinant -1
    n_samples : int, optional
        Number of samples of source to use when estimating distances.


    Returns
    -------
    np.array, np.array
        affine_matrix_rep : np.array of shape (4, 4)
            Affine transformation source -> target
        aligned : np.array of shape (n_source, 3)
            Aligned coordinates
    """
    target_centroid = np.mean(target, axis=0)
    target_inertia = get_inertia(target)
    target_eig = np.linalg.eigh(target_inertia)

    source_centroid = np.mean(source, axis=0)
    source_inertia = get_inertia(source)
    source_eig = np.linalg.eigh(source_inertia)

    flips = [np.diag([i,j,k]) for i, j, k in itertools.product(*(3*[[-1,1]]))]
    trafo_matrix_candidates = []
    tree = spatial.cKDTree(target)
    samples = source[np.random.randint(low=0, high=source.shape[0], size=min([n_samples, source.shape[0]])),:]
    distances = []
    for flip in flips:
        if shear:
            trafo_matrix = (source_eig.eigenvectors
                            @ np.diag(np.sqrt(target_eig.eigenvalues/source_eig.eigenvalues))
                            @ flip @ target_eig.eigenvectors.T)
        elif scale and not shear:
            scale_fact = np.sqrt(stats.gmean(target_eig.eigenvalues)/stats.gmean(source_eig.eigenvalues))
            trafo_matrix = scale_fact*source_eig.eigenvectors@flip@target_eig.eigenvectors.T
        elif not scale and not shear:
            trafo_matrix = source_eig.eigenvectors@flip@target_eig.eigenvectors.T
        if not improper and np.linalg.det(trafo_matrix) < 0:
            continue
        trafo_matrix = trafo_matrix.T
        trafo_matrix_candidates.append(trafo_matrix)
        trafo_translate = target_centroid - trafo_matrix@source_centroid
        aligned = samples@trafo_matrix.T + trafo_translate
        distances.append(np.mean(tree.query(aligned)[0]))
    trafo_matrix = trafo_matrix_candidates[np.argmin(distances)]
    print('inferred rotation/scale', trafo_matrix)
    trafo_translate = target_centroid - trafo_matrix@source_centroid
    aligned = source@trafo_matrix.T + trafo_translate
    affine_matrix_rep = package_affine_transformation(trafo_matrix, trafo_translate)
    
    print('inferred translation', trafo_translate)
    return affine_matrix_rep, aligned


def procrustes(source, target, scale=True):
    """
    Procrustes analysis, a similarity test for two data sets.

    Copied from scipy.spatial.procrustes, modified to return the transform
    as an affine matrix, and return the transformed source data in the original,
    non-normalized coordinates.

    Each input matrix is a set of points or vectors (the rows of the matrix).
    The dimension of the space is the number of columns of each matrix. Given
    two identically sized matrices, procrustes standardizes both such that:

    - tr(AA^T) = 1.
    - Both sets of points are centered around the origin.

    Procrustes then applies the optimal transform to the source matrix
    (including scaling/dilation, rotations, and reflections) to minimize the
    sum of the squares of the pointwise differences between the two input datasets.

    This function is not designed to handle datasets with different numbers of
    datapoints (rows).  If two data sets have different dimensionality
    (different number of columns), simply add columns of zeros to the smaller
    of the two.
    

    Parameters
    ----------
    source : array_like
        Matrix, n rows represent points in k (columns) space. The data from
        source will be transformed to fit the pattern in target.
    target : array_like
        Maxtrix, n rows represent points in k (columns) space. 
        target is the reference data. 
    scale : bool, default True
        Whether to allow scaling transformations

    Returns
    -------
    trafo_affine : array_like
        (4,4) array representing the affine transformation from source to target.
    aligned : array_like
        The orientation of source that best fits target.
    disparity : float
        np.linalg.norm(aligned-target, axis=1).mean()
    """
    mtx1 = np.array(target, dtype=np.float64, copy=True)
    mtx2 = np.array(source, dtype=np.float64, copy=True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    # translate all the data to the origin
    centroid1, centroid2 = (np.mean(mtx1, 0), np.mean(mtx2, 0))
    mtx1 -= centroid1
    mtx2 -= centroid2

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)
    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")
    mtx1 /= norm1
    mtx2 /= norm2
    # transform mtx2 to minimize disparity
    R, s = linalg.orthogonal_procrustes(mtx1, mtx2)
    mtx2 = np.dot(mtx2, R.T) * s

    # retranslate and scale
    aligned = norm1 * mtx2 + centroid1

    # measure the dissimilarity between the two datasets
    disparity = np.mean(np.linalg.norm(aligned-target, axis=1))

    # assemble the linear transformation
    if scale:
        trafo_matrix = (norm1/norm2)*s*R
    else:
        trafo_matrix = (norm1/norm2)*R
    trafo_translate = centroid1 - trafo_matrix@centroid2
    trafo_affine = package_affine_transformation(trafo_matrix, trafo_translate)
    return trafo_affine, aligned, disparity


def icp(source, target, initial=None, threshold=1e-4, max_iterations=20, scale=True, n_samples=1000):
    """
    Apply the iterative closest point algorithm to align point cloud a with
    point cloud b. Will only produce reasonable results if the
    initial transformation is roughly correct. Initial transformation can be
    found by applying Procrustes' analysis to a suitable set of landmark
    points (often picked manually), or by inertia+centroid based alignment,
    implemented in align_by_centroid_and_intertia.

    Parameters
    ----------
    source : (n,3) float
      Source points in space.
    target : (m,3) float or Trimesh
      Target points in space or mesh.
    initial : (4,4) float
      Initial transformation.
    threshold : float
      Stop when change in cost is less than threshold
    max_iterations : int
      Maximum number of iterations
    scale : bool, optional
      Whether to allow dilations. If False, orthogonal procrustes is used
    n_samples : int or None
        If not None, n_samples sample points are randomly chosen from source array for distance computation
    
    Returns
    ----------
    matrix : (4,4) float
      The transformation matrix sending a to b
    transformed : (n,3) float
      The image of a under the transformation
    cost : float
      The cost of the transformation
    """
    # initialize transform matrix
    total_matrix = np.eye(4) if initial is None else initial
    tree = spatial.cKDTree(target)
    # subsample and apply initial transformation
    samples = (source[np.random.randint(low=0, high=source.shape[0],
                                        size=min([n_samples, source.shape[0]])),:]
               if n_samples is not None else source[:])
    samples = samples@total_matrix[:3,:3].T + total_matrix[:3,-1]
    # start with infinite cost
    old_cost = np.inf
    # avoid looping forever by capping iterations
    for i in range(max_iterations):
        print('iteration', i, 'cost', old_cost) 
        # Find closest point in target to each point in sample and align
        closest = target[tree.query(samples, 1)[1]]
        matrix, samples, cost = procrustes(samples, closest, scale=scale)
        # update a with our new transformed points
        total_matrix = np.dot(matrix, total_matrix)
        if old_cost - cost < threshold:
            break
        else:
            old_cost = cost
    aligned = source@total_matrix[:3,:3].T + total_matrix[:3,-1]
    return total_matrix, aligned, cost


def combined_alignment(source, target, pre_align=True, shear=False, iterations=100):
    """Align source to target by combination of moment-of-intertia based aligment + ICP"""
    if pre_align:
        trafo_initial, _ = align_by_centroid_and_intertia(source, target,
                                                          scale=True, shear=shear, improper=False)
    else:
        trafo_initial = None
    trafo_icp, _, _ = icp(source, target, initial=trafo_initial,
                          threshold=1e-4, max_iterations=iterations,
                          scale=True, n_samples=5000)
    return trafo_icp


### Shrink-wrapping


def shrinkwrap_and_smooth(source_obj, target_obj, corrective_smooth_iter=0):
    """
    Applies a shrinkwrap modifier with target_obj to source_obj, 
    optionally adds a corrective smooth modifier, and applies all modifiers.

    Parameters:
    - source_obj: The source mesh object to be modified.
    - target_obj: The target mesh object for the shrinkwrap modifier.
    - corrective_smooth_iter: (Optional) Number of iterations for the corrective smooth modifier. 
      If 0, no corrective smooth is applied.

    Returns:
    - bpy.types.Object: The new modified mesh object.
    """
    # Ensure the objects are valid
    if source_obj.type != 'MESH' or target_obj.type != 'MESH':
        raise ValueError("Both source_obj and target_obj must be mesh objects.")

    # Store the currently active object
    original_active_obj = bpy.context.view_layer.objects.active

    # Add the first shrinkwrap modifier
    shrinkwrap_1 = source_obj.modifiers.new(name="Shrinkwrap", type='SHRINKWRAP')
    shrinkwrap_1.target = target_obj
    shrinkwrap_1.wrap_method = 'TARGET_PROJECT'

    # Add a corrective smooth modifier if requested
    for i in range(0, corrective_smooth_iter):
        corrective_smooth = source_obj.modifiers.new(name=f"Corrective Smooth {i}", type='CORRECTIVE_SMOOTH')
        corrective_smooth.iterations = 5
        corrective_smooth.scale = 0
        # Add a second shrinkwrap modifier after the corrective smooth
        shrinkwrap_2 = source_obj.modifiers.new(name=f"Shrinkwrap {i}", type='SHRINKWRAP')
        shrinkwrap_2.target = target_obj
        shrinkwrap_2.wrap_method = 'TARGET_PROJECT'

    # Apply all modifiers
    bpy.context.view_layer.objects.active = source_obj
    for modifier in source_obj.modifiers:
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    # Restore the original active object
    bpy.context.view_layer.objects.active = original_active_obj
    return source_obj


### Handling of mesh-associated array-data


def set_numpy_attribute(mesh, name, array):
    """Sets mesh[name] = array.
    
    Since Blender does not support adding arbitrary objects as attributes to meshes,
    the array is flattened, converted to a binary buffer, and saved as a tuple together with its shape.
    All arrays are converted to np.float32.
    """
    bytes, shape = (array.astype(np.float32).flatten().tobytes(), array.shape)
    mesh[name] = (bytes, shape)
    return None


def get_numpy_attribute(mesh, name):
    """Get array = mesh[name].
    
    Since Blender does not support adding arbitrary objects as attributes to meshes,
    the array is flattened, converted to a binary buffer, and saved as a tuple together with its shape.
    All arrays are converted to np.float32.
    """
    assert name in mesh, "Attribute not found"
    return np.frombuffer(mesh[name][0], dtype=np.float32).reshape(mesh[name][1])


def separate_selected_into_mesh_and_box(self, context):
    """
    Separate selected objects into mesh and box, representing 3D image data.
    
    If not exactly one mesh and one box (with attribute "3D_data") are selected,
    an error is raised.
    """
    n_data_selected = len([x for x in context.selected_objects if "3D_data" in x])
    n_mesh_selected = len([x for x in context.selected_objects if not "3D_data" in x])
    if not ((n_data_selected==1) and (n_mesh_selected==1)):
        self.report({'ERROR'}, "Select exactly one mesh and one 3D image (BoundingBox)!")
        return None, None
    box = [x for x in context.selected_objects if "3D_data" in x][0]
    obj = [x for x in context.selected_objects if not "3D_data" in x][0]
    if not obj or obj.type != 'MESH':
        self.report({'ERROR'}, "No mesh object selected!")
        return None, None
    return box, obj


### Operators defining the user interface of the add-on


class LoadTIFFOperator(Operator):
    """Load .tif file and resolution. Also creates a bounding box object."""
    bl_idname = "scene.load_tiff"
    bl_label = "Load TIFF File"

    def execute(self, context):
        file_path = bpy.path.abspath(context.scene.tissue_cartography_file)
        resolution = np.array(context.scene.tissue_cartography_resolution)
        self.report({'INFO'}, f"Resolution loaded: {resolution}")

        # Load TIFF file as a NumPy array
        if not (file_path.lower().endswith(".tiff") or file_path.lower().endswith(".tif")):
            self.report({'ERROR'}, "Selected file is not a TIFF")
            return {'CANCELLED'}
        try:
            data = tifffile.imread(file_path)
            if not len(data.shape) in [3,4]:
                self.report({'ERROR'}, "Selected TIFF must have 3 or 4 axes.")
                return {'CANCELLED'}
            if len(data.shape) == 3: # add singleton channel axis to single channel-data 
                data = data[np.newaxis]
            # ensure channel axis (assumed shortest axis) is 1st
            channel_axis = np.argmin(data.shape)
            data = np.moveaxis(data, channel_axis, 0)
            # permute axes
            axis_order = list(context.scene.tissue_cartography_axis_order)
            if not sorted(axis_order) == [0,1,2,3]:
                self.report({'ERROR'}, "Axis order must be a permutation of [0,1,2,3] (e.g. [3,0,1,2])")
                return {'CANCELLED'}
            data = data.transpose(axis_order)
            
            # display image shape in add-on
            context.scene.tissue_cartography_image_shape = str(data.shape[1:])
            context.scene.tissue_cartography_image_channels = data.shape[0]
            self.report({'INFO'}, f"TIFF file loaded with shape {data.shape}")
            # create a bounding box mesh to represent the data
            box = create_box(*(np.array(data.shape[1:])*resolution),
                             name=f"{Path(file_path).stem}_BoundingBox",
                             hide=False)
            box.display_type = 'WIRE'
            # attach the data to the box
            set_numpy_attribute(box, "resolution", resolution)
            set_numpy_attribute(box, "3D_data", data)
            
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load TIFF file: {e}")
            return {'CANCELLED'}

        return {'FINISHED'}


class LoadSegmentationTIFFOperator(Operator):
    """
    Load segmentation .tif file and resolution, and create a mesh from binary segmentation.
    
    Selecting a folder instead of a file batch processes all files in folder.
    """
    bl_idname = "scene.load_segmentation"
    bl_label = "Load Segmentation TIFF File"

    def execute(self, context):
        # Load resolution as a NumPy array
        resolution_array = np.array(context.scene.tissue_cartography_segmentation_resolution)
        input_path = Path(bpy.path.abspath(context.scene.tissue_cartography_segmentation_file))
        if input_path.is_dir():
            files_to_process = [f for f in input_path.iterdir() if f.is_file() and f.suffix in [".tif", ".tiff"]]
        elif input_path.is_file():
            files_to_process = [input_path]
        else:
            self.report({'ERROR'}, "Select a valid file or directory")
            return {'CANCELLED'}
        for file_path in files_to_process:
            if not file_path.suffix in [".tif", ".tiff"]:
                self.report({'ERROR'}, "Selected file is not a TIFF")
                return {'CANCELLED'}
            try:
                data = tifffile.imread(file_path)
                if len(data.shape) != 3:
                    self.report({'ERROR'}, "Data must be volumetric!")
                    return {'CANCELLED'}
                self.report({'INFO'}, f"TIFF file loaded with shape {data.shape}")
                # smooth and normalize the segmentation
                data = (data-data.min())/(data.max()-data.min())
                sigma = context.scene.tissue_cartography_segmentation_sigma
                data_smoothed = ndimage.gaussian_filter(data, sigma=sigma/resolution_array)
                # compute mesh using marching cubes, and convert to mesh
                verts, faces, _, _ = measure.marching_cubes(data_smoothed,
                                                            level=0.5, spacing=(1.0,1.0,1.0))
                verts = verts * resolution_array
                create_mesh_from_numpy(f"{Path(file_path).stem}_mesh", verts, faces)
                
            except Exception as e:
                self.report({'ERROR'}, f"Failed to load segmentation: {e}")
                return {'CANCELLED'}
        return {'FINISHED'}
    

class CreateProjectionOperator(Operator):
    """
    Create a cartographic projection.
    
    Select one mesh and one 3d-image ([...]_BoundingBox) to project 3d image data
    onto mesh surface.
    """
    bl_idname = "scene.create_projection"
    bl_label = "Create Projection"

    def execute(self, context):
        # Validate selected object and UV map
        box, obj = separate_selected_into_mesh_and_box(self, context)
        if box is None or obj is None:
            return {'CANCELLED'}
        # Ensure the object has a UV map
        if not obj.data.uv_layers:
            self.report({'ERROR'}, "The selected mesh does not have a UV map!")
            return {'CANCELLED'}
        # Parse offsets into a NumPy array
        offsets_str = context.scene.tissue_cartography_offsets
        try:
            offsets_array = np.array([float(x) for x in offsets_str.split(",") if x.strip()])
            if offsets_array.size == 0:
                offsets_array = np.array([0])
            self.report({'INFO'}, f"Offsets loaded: {offsets_array}")
        except ValueError as e:
            self.report({'ERROR'}, f"Invalid offsets input: {e}")
            return {'CANCELLED'}
        # set offsets as property
        set_numpy_attribute(obj, "projection_offsets", offsets_array)
        
        # Parse projection resolution
        projection_resolution = context.scene.tissue_cartography_projection_resolution
        self.report({'INFO'}, f"Using projection resolution: {projection_resolution}")

        # texture bake normals and world positions
        loop_uvs, loop_normals, loop_world_positions = get_uv_normal_world_per_loop(obj, filter_unique=True)
        
        baked_normals = bake_per_loop_values_to_uv(loop_uvs, loop_normals, 
                                                   image_resolution=projection_resolution)
        baked_normals = (baked_normals.T/np.linalg.norm(baked_normals.T, axis=0)).T
        baked_world_positions = bake_per_loop_values_to_uv(loop_uvs, loop_world_positions,
                                                           image_resolution=projection_resolution)
        # obtain UV layout and use it to get a mask
        uv_layout_path = str(Path(bpy.path.abspath("//")).joinpath(f'{obj.name}_UV_layout.png'))
        mask = get_uv_layout(obj, uv_layout_path, projection_resolution)
        baked_normals[~mask] = np.nan
        baked_world_positions[~mask] = np.nan
        
        # create a pullback
        box_world_inv = np.linalg.inv(np.array(box.matrix_world))
        baked_data = bake_volumetric_data_to_uv(get_numpy_attribute(box, "3D_data"),
                                                baked_world_positions, 
                                                get_numpy_attribute(box, "resolution"),
                                                baked_normals, normal_offsets=offsets_array,
                                                affine_matrix=box_world_inv)
        # set results as attributes of the mesh
        set_numpy_attribute(obj, "baked_data", baked_data)
        set_numpy_attribute(obj, "baked_normals", baked_normals)
        set_numpy_attribute(obj, "baked_world_positions", baked_world_positions)
        # create texture
        create_material_from_multilayer_array(obj, baked_data, material_name=f"ProjectedMaterial_{obj.name}")

        return {'FINISHED'}


class SaveProjectionOperator(Operator):
    """Save cartographic projection to disk"""
    bl_idname = "scene.save_projection"
    bl_label = "Save Projection"
    
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    
    def invoke(self, context, event):
        # Open file browser to choose the save location
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def execute(self, context):
        obj = context.active_object
        if not obj or "baked_data" not in obj:
            self.report({'ERROR'}, "No baked data found on the active object!")
            return {'CANCELLED'}
        
        # Get the baked data
        baked_data = get_numpy_attribute(obj, "baked_data")
        baked_normals = get_numpy_attribute(obj, "baked_normals")
        baked_world_positions = get_numpy_attribute(obj, "baked_world_positions")
        # Save the data to the chosen filepath
        try:
            tifffile.imwrite(self.filepath + "_BakedNormals.tif", baked_normals)
            tifffile.imwrite(self.filepath + "_BakedPositions.tif", baked_world_positions)
            tifffile.imwrite(self.filepath + "_BakedData.tif", baked_data.astype(np.float32),
                             metadata={'axes': 'ZCYX'}, imagej=True)
            self.report({'INFO'}, f"Cartographic projection saved to {self.filepath}")
        except Exception as e:
            self.report({'ERROR'}, f"Failed to save data: {str(e)}")
            return {'CANCELLED'}
        
        return {'FINISHED'}


class BatchProjectionOperator(Operator):
    """
    Batch-process cartographic projections.
    
    Select all meshes to process (in blender) and one 3d-image ([...]_BoundingBox)
    for resolution and relative position information. Further 3d .tiff files are read from
    Batch Process Input directory. Mesh names should match .tiff file names.
    
    """
    bl_idname = "scene.batch_projection"
    bl_label = "Create Projections (Batch Mode)"

    def execute(self, context):
        try:
            box = [x for x in context.selected_objects if "3D_data" in x][0]
        except IndexError:
            self.report({'ERROR'}, "Select one 3D image (BoundingBox) for resolution and position information!")
            return
        # get list of files
        batch_path = Path(bpy.path.abspath(context.scene.tissue_cartography_batch_directory))
        batch_out_path = Path(bpy.path.abspath(context.scene.tissue_cartography_batch_output_directory))
        batch_files = {f.stem: f for f in list(batch_path.iterdir()) if ((f.suffix in [".tif", ".tiff"]) and not "Baked" in f.stem)}
        # match files to selected meshes
        meshes_to_process = [obj for obj in context.selected_objects if obj != box]
        mesh_names = [obj.name for obj in meshes_to_process]
        matched = {obj.name: difflib.get_close_matches(obj.name, batch_files.keys(), n=1, cutoff=0.1)
                   for obj in context.selected_objects if obj != box}
        # parse axis order
        axis_order = list(context.scene.tissue_cartography_axis_order)
        if not sorted(axis_order) == [0,1,2,3]:
            self.report({'ERROR'}, "Axis order must be a permutation of [0,1,2,3] (e.g. [3,0,1,2])")
            return {'CANCELLED'}
        # parse offsets into a NumPy array
        offsets_str = context.scene.tissue_cartography_offsets
        try:
            offsets_array = np.array([float(x) for x in offsets_str.split(",") if x.strip()])
            if offsets_array.size == 0:
                offsets_array = np.array([0])
            self.report({'INFO'}, f"Offsets loaded: {offsets_array}")
        except ValueError as e:
            self.report({'ERROR'}, f"Invalid offsets input: {e}")
            return {'CANCELLED'}
        # Parse projection resolution
        projection_resolution = context.scene.tissue_cartography_projection_resolution
        self.report({'INFO'}, f"Using projection resolution: {projection_resolution}")
        # find box for position and resolution info
        
        for iobj, obj in enumerate(meshes_to_process):
            self.report({'INFO'}, f"Processing {iobj}/{len(meshes_to_process)}")
            if not obj.data.uv_layers:
                self.report({'ERROR'}, f"Mesh {obj.name} does not have a UV map!")
                return {'CANCELLED'}
            # set offsets as property
            set_numpy_attribute(obj, "projection_offsets", offsets_array)
            # find the matching file
            if len(matched[obj.name]) == 0:
                self.report({'ERROR'}, "No matching file found for {obj.name}!")
                return {'CANCELLED'}
            file_path = batch_files[matched[obj.name][0]]
            # load the 3D data
            try:
                data = tifffile.imread(file_path)
                if not len(data.shape) in [3,4]:
                    self.report({'INFO'}, f"Selected TIFF for {obj.name} must have 3 or 4 axes.")
                    return {'CANCELLED'}
                if len(data.shape) == 3: # add singleton channel axis to single channel-data 
                    data = data[np.newaxis]
                # ensure channel axis (assumed shortest axis) is 1st
                channel_axis = np.argmin(data.shape)
                data = np.moveaxis(data, channel_axis, 0)
                data = data.transpose(axis_order)
            except:
                self.report({'ERROR'}, f"Failed loading TIFF for {obj.name}")
                return {'CANCELLED'}
            # texture bake normals and world positions
            loop_uvs, loop_normals, loop_world_positions = get_uv_normal_world_per_loop(obj, filter_unique=True)
            
            baked_normals = bake_per_loop_values_to_uv(loop_uvs, loop_normals, 
                                                       image_resolution=projection_resolution)
            baked_normals = (baked_normals.T/np.linalg.norm(baked_normals.T, axis=0)).T
            baked_world_positions = bake_per_loop_values_to_uv(loop_uvs, loop_world_positions,
                                                               image_resolution=projection_resolution)
            # obtain UV layout and use it to get a mask
            uv_layout_path = str(Path(batch_out_path).joinpath(f'{obj.name}_UV_layout.png'))
            mask = get_uv_layout(obj, uv_layout_path, projection_resolution)
            baked_normals[~mask] = np.nan
            baked_world_positions[~mask] = np.nan
            # create a pullback
            box_world_inv = np.linalg.inv(np.array(box.matrix_world))
            baked_data = bake_volumetric_data_to_uv(data,
                                                    baked_world_positions, 
                                                    get_numpy_attribute(box, "resolution"),
                                                    baked_normals, normal_offsets=offsets_array,
                                                    affine_matrix=box_world_inv)
            # Save the data to the chosen filepath
            try:
                tifffile.imwrite(batch_out_path.joinpath(f"{obj.name}_BakedNormals.tif"), baked_normals)
                tifffile.imwrite(batch_out_path.joinpath(f"{obj.name}_BakedPositions.tif"), baked_world_positions)
                tifffile.imwrite(batch_out_path.joinpath(f"{obj.name}_BakedData.tif"), baked_data.astype(np.float32),
                                 metadata={'axes': 'ZCYX'}, imagej=True)
                self.report({'INFO'}, f"Cartographic projection saved for {obj.name}")
            except Exception as e:
                self.report({'ERROR'}, f"Failed to save data for {obj.name}: {str(e)}")
                return {'CANCELLED'}
            if bpy.context.scene.tissue_cartography_batch_create_materials:
                # set results as attributes of the mesh
                set_numpy_attribute(obj, "baked_data", baked_data)
                set_numpy_attribute(obj, "baked_normals", baked_normals)
                set_numpy_attribute(obj, "baked_world_positions", baked_world_positions)
                # create texture
                create_material_from_multilayer_array(obj, baked_data, material_name=f"ProjectedMaterial_{obj.name}")
        return {'FINISHED'}
    

class SlicePlaneOperator(Operator):
    """Create a slice plane along the selected axis with texture from 3D data"""
    bl_idname = "scene.create_slice_plane"
    bl_label = "Create Slice Plane"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        # Get the 3D data array from the selected box
        box = context.active_object
        if not box or not "3D_data" in box:
            self.report({'ERROR'}, "Select exactly a 3D image (BoundingBox)!")
            return {'CANCELLED'}
        data = get_numpy_attribute(box, "3D_data")
        
        resolution = get_numpy_attribute(box, "resolution")
        if not isinstance(data, np.ndarray) or data.ndim != 4:
            self.report({'ERROR'}, "Invalid 3D data array.")
            return {'CANCELLED'}
        if context.scene.tissue_cartography_slice_channel >= data.shape[0]:
            self.report({'ERROR'}, f"Channel {context.scene.tissue_cartography_slice_channel} is out of bounds for the data array.")
            return {'CANCELLED'}

        length, width, height = (np.array(data.shape[1:]) * resolution)
        slice_plane = create_slice_plane(length, width, height, axis=context.scene.tissue_cartography_slice_axis,
                                         position=context.scene.tissue_cartography_slice_position)
        slice_plane.name = f"{slice_plane.name}_{box.name}"
        # set matrix world
        slice_plane.matrix_world = box.matrix_world
                                         
        slice_img = get_slice_image(data, resolution, axis=context.scene.tissue_cartography_slice_axis,
                                    position=context.scene.tissue_cartography_slice_position)
        slice_img = normalize_quantiles(slice_img, quantiles=(0.01, 0.99),
                                        channel_axis=0, clip=True, data_type=None)     
        create_material_from_array(slice_plane, slice_img[context.scene.tissue_cartography_slice_channel], material_name=f"SliceMaterial_{box.name}_{context.scene.tissue_cartography_slice_axis}_{context.scene.tissue_cartography_slice_position}")  
        return {'FINISHED'}


class VertexShaderInitializeOperator(Operator):
    """Initialize vertex shader for a selected mesh. Colors mesh vertices according to 
    3D image intensity from selected BoundingBox."""
    bl_idname = "scene.initialize_vertex_shader"
    bl_label = "Initialize Vertex Shader"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        # create global dict to hold interpolator objects
        if not hasattr(bpy.types.Scene, "tissue_cartography_interpolators"):
            bpy.types.Scene.tissue_cartography_interpolators = dict()
        # get the selected mesh and bounding box
        box, obj = separate_selected_into_mesh_and_box(self, context)
        if box is None or obj is None:
            return {'CANCELLED'}
        # Get the 3D data array from the box object
        data = get_numpy_attribute(box, "3D_data")
        resolution = get_numpy_attribute(box, "resolution")
     
        if not isinstance(data, np.ndarray) or data.ndim != 4:
            self.report({'ERROR'}, "Invalid 3D data array.")
            return {'CANCELLED'}
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "No mesh object selected!")
            return {'CANCELLED'}
        if context.scene.tissue_cartography_vertex_shader_channel >= data.shape[0]:
            self.report({'ERROR'}, f"Channel {context.scene.tissue_cartography_vertex_shader_channel} is out of bounds for the data array.")
            return {'CANCELLED'}
        # need to compute coordinates relative to matrix_world of box I think
        set_numpy_attribute(obj, "box_world_inv_vertex_shader",
                            np.array(box.matrix_world.inverted()))
        bpy.types.Scene.tissue_cartography_interpolators[obj.name] = get_image_to_vertex_interpolator(obj, data, resolution)
        box_inv = mathutils.Matrix(get_numpy_attribute(obj, 
                                   "box_world_inv_vertex_shader"))
        positions = np.array([box_inv@obj.matrix_world@(v.co + context.scene.tissue_cartography_vertex_shader_offset*v.normal)
                              for v in obj.data.vertices])
        intensities = bpy.types.Scene.tissue_cartography_interpolators[obj.name][context.scene.tissue_cartography_vertex_shader_channel](positions)
        colors = np.stack(3*[intensities,], axis=1)
        
        assign_vertex_colors(obj, colors)
        create_vertex_color_material(obj, material_name=f"VertexColorMaterial_{obj.name}")

        return {'FINISHED'}


class VertexShaderRefreshOperator(Operator):
    """Refresh vertex colors for a selected mesh. Colors mesh vertices according to 
    3D image intensity."""
    bl_idname = "scene.refresh_vertex_shader"
    bl_label = "Refresh Vertex Shader"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        interpolator_dict = getattr(context.scene, "tissue_cartography_interpolators")
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "No mesh object selected!")
            return {'CANCELLED'}
        if interpolator_dict is None or obj.name not in interpolator_dict:
            self.report({'ERROR'}, f"Vertex shader not initialized.")
            return {'CANCELLED'}
        if context.scene.tissue_cartography_vertex_shader_channel >= len(interpolator_dict[obj.name]):
            self.report({'ERROR'}, f"Channel {context.scene.tissue_cartography_vertex_shader_channel} is out of bounds for the data array.")
        box_inv = mathutils.Matrix(get_numpy_attribute(obj, "box_world_inv_vertex_shader"))
        positions = np.array([box_inv@obj.matrix_world@(v.co + context.scene.tissue_cartography_vertex_shader_offset*v.normal)
                              for v in obj.data.vertices])
        intensities = interpolator_dict[obj.name][context.scene.tissue_cartography_vertex_shader_channel](positions)
        colors = np.stack(3*[intensities,], axis=1)
        assign_vertex_colors(obj, colors)

        return {'FINISHED'}


class AlignOperator(Operator):
    """Align active and selected meshes by rotation, translation, and scaling."""
    bl_idname = "scene.align"
    bl_label = "Align Selected To Active Mesh"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        
        if context.scene.tissue_cartography_align_type == "selected":
            target_mesh = context.active_object
            for source_mesh in [x for x in context.selected_objects if not x==target_mesh]:
                self.report({'INFO'}, f"Aligning: {source_mesh.name} to {target_mesh.name}")
                if target_mesh.type != 'MESH'  or source_mesh.type != 'MESH':
                    self.report({'ERROR'}, "Selected object(s) is not a mesh.")
                    return {'CANCELLED'}
                # Get the 3D coordinates from the meshes
                target = np.array([target_mesh.matrix_world@v.co for v in target_mesh.data.vertices])
                source = np.array([source_mesh.matrix_world@v.co for v in source_mesh.data.vertices])
                trafo_matrix = combined_alignment(source, target,
                                                  pre_align=context.scene.tissue_cartography_prealign,
                                                  shear=context.scene.tissue_cartography_prealign_shear,
                                                  iterations=context.scene.tissue_cartography_align_iter)
                source_mesh.matrix_world = mathutils.Matrix(trafo_matrix)@ source_mesh.matrix_world
        elif context.scene.tissue_cartography_align_type == "active":
            source_mesh = context.active_object
            for target_mesh in [x for x in context.selected_objects if not x==source_mesh]:
                self.report({'INFO'}, f"Aligning: {source_mesh.name} to {target_mesh.name}")
                if target_mesh.type != 'MESH'  or source_mesh.type != 'MESH':
                    self.report({'ERROR'}, "Selected object(s) is not a mesh.")
                    return {'CANCELLED'}
                # Get the 3D coordinates from the meshes and compute alignment
                target = np.array([target_mesh.matrix_world@v.co for v in target_mesh.data.vertices])
                source = np.array([source_mesh.matrix_world@v.co for v in source_mesh.data.vertices])
                trafo_matrix = combined_alignment(source, target,
                                                  pre_align=context.scene.tissue_cartography_prealign,
                                                  shear=context.scene.tissue_cartography_prealign_shear,
                                                  iterations=context.scene.tissue_cartography_align_iter)
                # copy source mesh
                source_mesh_copied = source_mesh.copy()
                source_mesh_copied.data = source_mesh.data.copy()
                bpy.context.collection.objects.link(source_mesh_copied)
                source_mesh_copied.name = f"{target_mesh.name}_aligned" 
                source_mesh_copied.matrix_world = mathutils.Matrix(trafo_matrix)@ source_mesh.matrix_world
        return {'FINISHED'}


class ShrinkwrapOperator(Operator):
    """Copy and shrink-wrap active mesh to selected meshes."""
    bl_idname = "scene.shrinkwrap"
    bl_label = "Shrink-Wrap Active to Selected"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        mode = context.scene.tissue_cartography_shrinkwarp_iterative
        source_mesh = context.active_object
        targets = sorted([x for x in context.selected_objects if not x==source_mesh], key=lambda x: x.name)
        if mode == "backward":
            targets = targets[::-1]
        for target_mesh in targets:
            self.report({'INFO'}, f"Aligning: {source_mesh.name} to {target_mesh.name}")
            if target_mesh.type != 'MESH'  or source_mesh.type != 'MESH':
                self.report({'ERROR'}, "Selected object(s) is not a mesh.")
                return {'CANCELLED'}
            # rigid alignment
            target = np.array([target_mesh.matrix_world@v.co for v in target_mesh.data.vertices])
            source = np.array([source_mesh.matrix_world@v.co for v in source_mesh.data.vertices])
            trafo_matrix = combined_alignment(source, target,
                                              pre_align=context.scene.tissue_cartography_prealign,
                                              shear=context.scene.tissue_cartography_prealign_shear,
                                              iterations=context.scene.tissue_cartography_align_iter)
            # copy source mesh
            source_mesh_copied = source_mesh.copy()
            source_mesh_copied.data = source_mesh.data.copy()
            bpy.context.collection.objects.link(source_mesh_copied)
            source_mesh_copied.matrix_world = mathutils.Matrix(trafo_matrix)@ source_mesh.matrix_world
            source_mesh_copied.name = f"{target_mesh.name}_wrapped" 
            # shrink-wrap
            shrinkwrap_and_smooth(source_mesh_copied, target_mesh,
                                  corrective_smooth_iter=context.scene.tissue_cartography_shrinkwarp_smooth)
            # data transfer modifier to copy UV map from wrapped to target
            data_transfer = target_mesh.modifiers.new(name="DataTransfer", type='DATA_TRANSFER')
            data_transfer.object = source_mesh_copied
            data_transfer.use_loop_data = True
            data_transfer.data_types_loops = {'UV'}
            data_transfer.loop_mapping = 'POLYINTERP_NEAREST'
            # apply
            original_active_obj = bpy.context.view_layer.objects.active
            bpy.context.view_layer.objects.active = target_mesh
            bpy.ops.object.datalayout_transfer(modifier="DataTransfer")
            bpy.ops.object.modifier_apply(modifier="DataTransfer")
            bpy.context.view_layer.objects.active = original_active_obj
                                  
            if mode in ["forward", "backward"]:
                source_mesh = source_mesh_copied
        return {'FINISHED'}


class HelpPopupOperator(Operator):
    """Open help window."""
    bl_idname = "scene.help_popup"
    bl_label = "Tissue Cartography Help"

    def execute(self, context):
        url = "https://nikolas-claussen.github.io/blender-tissue-cartography/Tutorials/03_blender_addon_tutorial.html"
        bpy.ops.wm.url_open(url=url)
        return {'FINISHED'}
        
        
class TissueCartographyPanel(Panel):
    """Class defining layout of user interface (buttons, inputs, etc.)"""
    bl_label = "Tissue Cartography"
    bl_idname = "SCENE_PT_tissue_cartography"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "scene"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        layout.prop(scene, "tissue_cartography_file")
        layout.prop(scene, "tissue_cartography_resolution")
        layout.prop(scene, "tissue_cartography_axis_order")
        layout.operator("scene.load_tiff", text="Load .tiff file")
        layout.label(text=f"Loaded Image Shape: {scene.tissue_cartography_image_shape}. Loaded Image Channels: {scene.tissue_cartography_image_channels}")
        layout.separator()
        
        layout.prop(scene, "tissue_cartography_segmentation_file")
        row_segmentation = layout.row()
        row_segmentation.prop(scene, "tissue_cartography_segmentation_resolution")
        row_segmentation.prop(scene, "tissue_cartography_segmentation_sigma")
        layout.operator("scene.load_segmentation", text="Get mesh(es) from binary segmentation .tiff file(s)")
        layout.separator()
        
        row_slice = layout.row()
        row_slice.prop(scene, "tissue_cartography_slice_axis")
        row_slice.prop(scene, "tissue_cartography_slice_position")
        row_slice.prop(scene, "tissue_cartography_slice_channel")
        layout.operator("scene.create_slice_plane", text="Create slice plane")
        layout.separator()
        
        row_vertex = layout.row()
        row_vertex.prop(scene, "tissue_cartography_vertex_shader_offset")
        row_vertex.prop(scene, "tissue_cartography_vertex_shader_channel")
        row_vertex2 = layout.row()
        row_vertex2.operator("scene.initialize_vertex_shader", text="Initialize vertex shading")
        row_vertex2.operator("scene.refresh_vertex_shader", text="Refresh vertex shading")
        layout.separator()
        
        row_projection = layout.row()
        row_projection.prop(scene, "tissue_cartography_offsets")
        row_projection.prop(scene, "tissue_cartography_projection_resolution")
        row_projection2 = layout.row()
        row_projection2.operator("scene.create_projection", text="Create Projection")
        row_projection2.operator("scene.save_projection", text="Save Projection")
        layout.separator()
        
        row_batch = layout.row()
        row_batch.prop(scene, "tissue_cartography_batch_directory")
        row_batch.prop(scene, "tissue_cartography_batch_output_directory")
        row_batch.prop(scene, "tissue_cartography_batch_create_materials")
        layout.operator("scene.batch_projection", text="Batch Process And Save")
        layout.separator()
        
        row_align = layout.row()
        row_align.prop(scene, "tissue_cartography_prealign")
        row_align.prop(scene, "tissue_cartography_prealign_shear")
        row_align.prop(scene, "tissue_cartography_align_type")
        row_align.prop(scene, "tissue_cartography_align_iter")
        layout.operator("scene.align", text="Align Meshes")
        layout.separator()
        
        row_shrinkwrap = layout.row()
        row_shrinkwrap.prop(scene, "tissue_cartography_shrinkwarp_smooth")
        row_shrinkwrap.prop(scene, "tissue_cartography_shrinkwarp_iterative")
        layout.operator("scene.shrinkwrap", text="Shrinkwrap Meshes (Active To Selected)")
        layout.separator()
        
        layout.operator("scene.help_popup", text="Show help", icon='HELP')
    
    

def register():
    """Add the add-on to the blender user interface"""
    bpy.utils.register_class(TissueCartographyPanel)
    bpy.utils.register_class(LoadTIFFOperator)
    bpy.utils.register_class(LoadSegmentationTIFFOperator)
    bpy.utils.register_class(CreateProjectionOperator)
    bpy.utils.register_class(SaveProjectionOperator)
    bpy.utils.register_class(BatchProjectionOperator)
    bpy.utils.register_class(SlicePlaneOperator)
    bpy.utils.register_class(VertexShaderInitializeOperator)
    bpy.utils.register_class(VertexShaderRefreshOperator)
    bpy.utils.register_class(AlignOperator)
    bpy.utils.register_class(ShrinkwrapOperator)
    bpy.utils.register_class(HelpPopupOperator)
    
    bpy.types.Scene.tissue_cartography_file = StringProperty(
        name="File Path",
        description="Path to the TIFF file",
        subtype='FILE_PATH',
    )
    bpy.types.Scene.tissue_cartography_resolution = FloatVectorProperty(
        name="x/y/z Resolution (µm)",
        description="Resolution in microns along x, y, z axes",
        size=3,
        default=(1.0, 1.0, 1.0),
    )
    bpy.types.Scene.tissue_cartography_axis_order= IntVectorProperty(
        name="Axis order",
        description="Permute axes after loading image. Use if loaded image shape appears incorrect. (0, 1, 2, 3) =  no permutation. Channel axis should be first axis!",
        size=4,
        default=(0, 1, 2, 3),
        min=0,
        max=3,
    )
    bpy.types.Scene.tissue_cartography_image_channels = IntProperty(
        name="Image Channels",
        description="Channels of the loaded image (read-only)",
        default=0,
        min=0,
    )
    bpy.types.Scene.tissue_cartography_image_shape = StringProperty(
        name="Image Shape",
        description="Shape of the loaded image (read-only)",
        default="Not loaded"
    )
    bpy.types.Scene.tissue_cartography_segmentation_file = StringProperty(
        name="Segmentation File Path",
        description="Path to the segmentation TIFF file. Should have values between 0-1. Selecting a folder instead of a single file will batch-process the full folder.",
        subtype='FILE_PATH',
    )
    bpy.types.Scene.tissue_cartography_segmentation_resolution = FloatVectorProperty(
        name="Segmentation x/y/z Resolution (µm)",
        description="Resolution of segmentation in microns along x, y, z axes",
        size=3,
        default=(1.0, 1.0, 1.0),
    )
    bpy.types.Scene.tissue_cartography_segmentation_sigma = FloatProperty(
        name="Segmentation Smoothing (µm)",
        description="Smothing kernel for extracting mesh from segmentation, in µm",
        default=0,
        min=0
    ) 
    bpy.types.Scene.tissue_cartography_slice_axis = EnumProperty(
        name="Slice Axis",
        description="Choose an axis",
        items=[('x', "X-Axis", "Align to the X axis"),
               ('y', "Y-Axis", "Align to the Y axis"),
               ('z', "Z-Axis", "Align to the Z axis")],
        default='x'
    )
    bpy.types.Scene.tissue_cartography_slice_position = FloatProperty(
        name="Slice Position (µm)",
        description="Position along the selected axis in µm",
        default=0
    )
    bpy.types.Scene.tissue_cartography_slice_channel = IntProperty(
        name="Slice Channel",
        description="Channel for slice plane",
        default=0,
        min=0,
    )
    bpy.types.Scene.tissue_cartography_vertex_shader_offset = FloatProperty(
        name="Vertex Shader Normal Offset (µm)",
        description="Normal offse for vertex shading.",
        default=0,
    )
    bpy.types.Scene.tissue_cartography_vertex_shader_channel = IntProperty(
        name="Vertex Shader Channel",
        description="Channel for vertex shading.",
        default=0,
        min=0,
    )
    bpy.types.Scene.tissue_cartography_offsets = StringProperty(
        name="Normal Offsets (µm)",
        description="Comma-separated list of floats for multilayer projection offsets",
        default="0",
    )
    bpy.types.Scene.tissue_cartography_projection_resolution = IntProperty(
        name="Projection Format (Pixels)",
        description="Resolution for the projection (e.g., 1024 for 1024x1024 pixels)",
        default=1024,
        min=1,
    )
    
    bpy.types.Scene.tissue_cartography_batch_directory = StringProperty(
        name="Batch Process Input Directory",
        description="Path to TIFF files directory",
        subtype='DIR_PATH',
    )
    bpy.types.Scene.tissue_cartography_batch_output_directory = StringProperty(
        name="Batch Process Output Directory",
        description="Path to TIFF files directory",
        subtype='DIR_PATH',
    )
    bpy.types.Scene.tissue_cartography_batch_create_materials = BoolProperty(
        name="Create materials",
        description="Enable or disable creating materials with projected texture in batch mode. Enabling can result in large .blend files.",
        default=True
    )
    bpy.types.Scene.tissue_cartography_prealign = BoolProperty(
        name="Pre-align?",
        description="Enable or disable pre-alignment. Do not use if the two meshes are already closely aligned.",
        default=True
    )
    bpy.types.Scene.tissue_cartography_prealign_shear = BoolProperty(
        name="Allow shear",
        description="Allow shear transformation during alignment.",
        default=True
    )
    bpy.types.Scene.tissue_cartography_align_type = EnumProperty(
        name="Align Mode",
        description="Choose an axis",
        items=[('selected', "Selected to Active", "Align selected meshes to active mesh."),
               ('active', "Active to Selected", "Align active mesh to selected meshe (creates copies of active mesh).")],
        default='selected'
    )
    bpy.types.Scene.tissue_cartography_align_iter = IntProperty(
        name="Iterations",
        description="ICP iterations during alignment.",
        default=100,
        min=1,
    )
    bpy.types.Scene.tissue_cartography_shrinkwarp_smooth = IntProperty(
        name="Shrinkwrap Corrective Smooth",
        description="Corrective smooth iterations during shrink-wrapping.",
        default=10,
        min=0,
    )
    bpy.types.Scene.tissue_cartography_shrinkwarp_iterative = EnumProperty(
        name="Shrinkwrap Mode",
        description="Choose an axis",
        items=[('one-to-all', "One-To-All", "Shrink-wrap active mesh to each selected individually"),
               ('forward', "Iterative Forward", "Shrink-wrap active mesh to selected meshes iteratively, starting with alpha-numerically first"),
               ('backward', "Iterative Backward", "Shrink-wrap active mesh to selected meshes iteratively, starting with alpha-numerically last")],
        default='one-to-all'
    )

def unregister():
    bpy.utils.unregister_class(TissueCartographyPanel)
    bpy.utils.unregister_class(LoadTIFFOperator)
    bpy.utils.unregister_class(LoadSegmentationTIFFOperator)
    bpy.utils.unregister_class(CreateProjectionOperator)
    bpy.utils.unregister_class(BatchProjectionOperator)
    bpy.utils.unregister_class(SaveProjectionOperator)
    bpy.utils.unregister_class(SlicePlaneOperator)
    bpy.utils.unregister_class(VertexShaderInitializeOperator)
    bpy.utils.unregister_class(VertexShaderRefreshOperator)
    bpy.utils.unregister_class(AlignOperator)
    bpy.utils.unregister_class(ShrinkwrapOperator)
    bpy.utils.unregister_class(HelpPopupOperator)

    del bpy.types.Scene.tissue_cartography_file 
    del bpy.types.Scene.tissue_cartography_resolution
    del bpy.types.Scene.tissue_cartography_axis_order
    del bpy.types.Scene.tissue_cartography_image_shape
    del bpy.types.Scene.tissue_cartography_segmentation_file
    del bpy.types.Scene.tissue_cartography_segmentation_resolution 
    del bpy.types.Scene.tissue_cartography_segmentation_sigma
    del bpy.types.Scene.tissue_cartography_offsets 
    del bpy.types.Scene.tissue_cartography_projection_resolution 
    del bpy.types.Scene.tissue_cartography_slice_axis 
    del bpy.types.Scene.tissue_cartography_slice_position 
    del bpy.types.Scene.tissue_cartography_slice_channel 
    del bpy.types.Scene.tissue_cartography_vertex_shader_offset 
    del bpy.types.Scene.tissue_cartography_vertex_shader_channel
    del bpy.types.Scene.tissue_cartography_prealign 
    del bpy.types.Scene.tissue_cartography_prealign_shear
    del bpy.types.Scene.tissue_cartography_align_iter
    del bpy.types.Scene.tissue_cartography_align_type
    del bpy.types.Scene.tissue_cartography_batch_directory
    del bpy.types.Scene.tissue_cartography_batch_output_directory
    del bpy.types.Scene.tissue_cartography_batch_create_materials
    del bpy.types.Scene.tissue_cartography_shrinkwarp_smooth
    del bpy.types.Scene.tissue_cartography_shrinkwarp_iterative
    
    del bpy.types.Scene.tissue_cartography_interpolators

### Run the add-on


if __name__ == "__main__":
    register()