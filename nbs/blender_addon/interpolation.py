import bpy
import numpy as np
import tifffile
from scipy import interpolate
from pathlib import Path

# Main tissue cartography functions - baking normals and world positions to UV, interpolating 3d data to UV using those bakes

def load_png(image_path):
    """Load .png into numpy array."""
    image = bpy.data.images.load(image_path)
    width, height = image.size
    pixels = np.array(image.pixels[:], dtype=np.float32)
    return pixels.reshape((height, width, -1))

def get_uv_layout(uv_layout_path, image_resolution):
    """Get UV layout mask for currently active object and save UV layout to disk"""
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.uv.export_layout(filepath=uv_layout_path, size=(image_resolution, image_resolution), opacity=1)
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
    
def bake_volumetric_data_to_uv(image, baked_world_positions, resolution, baked_normals, normal_offsets=(0,)):
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
        
    Returns
    -------
    aked_data : np.array of shape (n_channels, n_layers, image_resolution, image_resolution)
        Multi-layer 3d volumetric data baked onto UV.
    """
    x, y, z = [np.arange(ni) for ni in image.shape[1:]]
    baked_data = []
    for o in normal_offsets:
        baked_layer_data = np.stack([interpolate.interpn((x, y, z), channel,
                                     (baked_world_positions+o*baked_normals)/resolution,
                                     method="linear", bounds_error=False) for channel in image])
        baked_data.append(baked_layer_data)
    baked_data = np.stack(baked_data, axis=1)
    return baked_data

###### Script
    
image_resolution = 256
loop_uvs, loop_normals, loop_world_positions = get_uv_normal_world_per_loop(bpy.context.object, filter_unique=True)
baked_normals = bake_per_loop_values_to_uv(loop_uvs, loop_normals, image_resolution=image_resolution)
baked_normals = (baked_normals.T/np.linalg.norm(baked_normals.T, axis=0)).T
baked_world_positions = bake_per_loop_values_to_uv(loop_uvs, loop_world_positions, image_resolution=image_resolution)

# obtain UV layout and use it to get a mask
uv_layout_path = str(Path(bpy.path.abspath("//")).joinpath('UV_layout.png'))
mask = get_uv_layout(uv_layout_path, image_resolution)
baked_normals[~mask] = np.nan
baked_world_positions[~mask] = np.nan

# create a pullback
image_path = "/home/nikolas/Documents/UCSB/streichan/numerics/code/python_code/jupyter_notebooks/blender-tissue-cartography/nbs/Tutorials/drosophila_example/Drosophila_CAAX-mCherry.tif"
image = tifffile.imread(image_path)
image = image[np.newaxis]
baked_data = bake_volumetric_data_to_uv(image, baked_world_positions, np.array([1,1,1]), baked_normals, normal_offsets=(0,1))

# set as global variables and save to disk
bpy.types.Scene.tissue_cartography_image = image
bpy.types.Scene.tissue_cartography_baked_normals = baked_normals
bpy.types.Scene.tissue_cartography_baked_world_positions = baked_world_positions
bpy.types.Scene.tissue_cartography_baked_data = baked_data
tifffile.imwrite(Path(bpy.path.abspath("//")).joinpath('NormalMapBarycentric.tif'), baked_normals)
tifffile.imwrite(Path(bpy.path.abspath("//")).joinpath('PositionMapBarycentric.tif'), baked_world_positions)
tifffile.imwrite(Path(bpy.path.abspath("//")).joinpath('BakedData.tif'), baked_data)
