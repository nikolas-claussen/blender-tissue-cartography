# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/02_cartographic_interpolation.ipynb.

# %% auto 0
__all__ = ['get_uv_layout_mask_mask', 'interpolate_3d_to_uv', 'interpolate_volumetric_data_to_uv',
           'interpolate_volumetric_data_to_uv_multilayer', 'create_cartographic_projections']

# %% ../nbs/02_cartographic_interpolation.ipynb 1
import numpy as np
from skimage import transform
from scipy import interpolate
import matplotlib as mpl
import matplotlib.pyplot as plt
from . import io as tcio

# %% ../nbs/02_cartographic_interpolation.ipynb 28
def get_uv_layout_mask_mask(mesh, uv_grid_steps=256):
    """
    Get a layout mask of the UV square: 1 where the UV square is covered by the unwrapped mesh, 0 outside.
    
    Based on a questionable matplotlib hack.
    
    Parameters
    ----------
    mesh : dict
        Mesh, as dict with entries "vertices", "texture_vertices", "normals", and "faces"
    uv_grid_steps : int, default 256
        Size of UV grid. Determines resolution of result.
    
    Returns
    -------
    uv_mask : np.array of shape (uv_grid_steps, uv_grid_steps)
        Mask of the part of the UV square covered by the unwrapped mesh
    """
    valid_faces = [[v[1] for v in fc] for fc in mesh["faces"] if not np.isnan(list(tcio.flatten(fc))).any()]
    polygons = mpl.collections.PatchCollection([mpl.patches.Polygon([mesh["texture_vertices"][v] for v in fc])
                                                for fc in valid_faces], color="black")
    fig = plt.figure(figsize=(1,1), dpi=uv_grid_steps, frameon=False)
    ax = plt.gca()
    ax.add_collection(polygons)
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    fig.canvas.draw()
    uv_mask = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).astype(float)
    uv_mask = 1-(uv_mask.reshape(fig.canvas.get_width_height()[::-1] + (4,))[...,0] > 0)
    plt.close()
    
    return uv_mask.astype(bool)

# %% ../nbs/02_cartographic_interpolation.ipynb 33
def interpolate_3d_to_uv(matched_texture_vertices, matched_vertices_or_normals, uv_mask=None, uv_grid_steps=256):
    """
    Interpolate 3d mesh coordinates or mesh normals onto UV square.
    
    Assumes the map $x,y,z \mapsto u,v$ to be invertible. This is not guaranteed - you can create overlapping UV 
    coordinates in blender. 
    
    Parameters
    ----------
    matched_vertices_or_normals : np.array of shape (n_matched, 3)
        Vertex 3d coordinates or normals, matched to UV coordinates
    matched_texture_vertices : np.array of shape (n_matched, 2)
        Matched texture vertices. Will be mapped back to [0, 1]^2!
    uv_mask : None or np.array of shape (uv_grid_steps, uv_grid_steps) of dtype bool
        Mask of covered part of the UV square. If provided, interpolation results are set to np.nan outside the
        covered region. If None, no masking takes place. No masking may result in spurious results
        in the part of the UV square not covered by the unwrapped mesh.
    uv_grid_steps : int, default 256
        Size of UV grid. Determines resolution of result.

    Returns
    -------
    interpolated_3d : np.array of shape (uv_grid_steps, uv_grid_steps, 3)
        3d positions or normals across [0,1]^2 UV grid, with uniform step size. UV positions that don't
        correspond to any value are set to np.nan.
        
    """
    matched_texture_vertices %= 1
    u, v = 2*[np.linspace(0,1, uv_grid_steps),]
    U, V = np.meshgrid(u, v)
    interpolated_3d = np.stack([interpolate.griddata(matched_texture_vertices, x, (U, V), method='linear')
                                for x in matched_vertices_or_normals.T], axis=-1)
    interpolated_3d = interpolated_3d[::-1] # stupid axis convention issue!!
    if uv_mask is not None:
        interpolated_3d[~uv_mask,:] = np.nan
    return interpolated_3d

# %% ../nbs/02_cartographic_interpolation.ipynb 34
def interpolate_volumetric_data_to_uv(image, interpolated_3d_positions, resolution, uv_mask=None):
    """ 
    Interpolate volumetric image data onto UV coordinate grid.
    
    Uses 3d positions corresponding to each UV grid point as computed by interpolate_3d_to_uv.
    3d coordinates (in microns) are converted into image coordinates via the scaling factor.
    
    Parameters
    ----------
    image : 4d np.array
        Image, axis 0  is assumed to be the channel axis
    interpolated_3d_positions : np.array of shape (uv_grid_steps, uv_grid_steps, 3)
        3d positions across [0,1]^2 UV grid, with uniform step size. UV positions that don't correspond to 
        any value are set to np.nan.
    resolution : np.array of shape (3,)
        Resolution in pixels/microns for each of the three spatial axes.
    uv_mask : None or np.array of shape (uv_grid_steps, uv_grid_steps) of dtype bool
        Mask of covered part of the UV square. If provided, interpolation results are set to np.nan outside the
        covered region. If None, no masking takes place. No masking may result in spurious results
        in the part of the UV square not covered by the unwrapped mesh.

    
    Returns
    -------
    interpolated_data : np.array of shape (n_channels, uv_grid_steps, uv_grid_steps)
        3d volumetric data interpolated onto UV grid.
    
    """
    x, y, z = [np.arange(ni) for ni in image.shape[1:]]
    interpolated_data = np.stack([interpolate.interpn((x, y, z), channel, interpolated_3d_positions/resolution,
                                  method="linear", bounds_error=False) for channel in image])
    if uv_mask is not None:
        interpolated_data[:,~uv_mask] = np.nan
    
    return interpolated_data

# %% ../nbs/02_cartographic_interpolation.ipynb 37
def interpolate_volumetric_data_to_uv_multilayer(image, interpolated_3d_positions, interpolated_normals,
                                                 normal_offsets, resolution, uv_mask=None):
    """ 
    Multilayer-interpolate volumetric image data onto UV coordinate grid.
    
    Uses 3d positions corresponding to each UV grid point as computed by interpolate_3d_to_uv.
    3d coordinates (in microns) are converted into image coordinates via the scaling factor.
    
    Generates multiple "layers" by shifting surface along its normals.
    
    Parameters
    ----------
    image : 4d np.array
        Image, axis 0  is assumed to be the channel axis
    interpolated_3d_positions : np.array of shape (uv_grid_steps, uv_grid_steps, 3)
        3d positions across [0,1]^2 UV grid, with uniform step size. UV positions that don't correspond to 
        any value are set to np.nan.
    interpolated_normals : np.array of shape (uv_grid_steps, uv_grid_steps, 3)
        3d normals across [0,1]^2 UV grid, with uniform step size. UV positions that don't correspond to 
        any value are set to np.nan. Normal vectors will be automatically normalized.
    normal_offsets : np.array of shape (n_layers,)
        Offsets along normal direction, in same units as interpolated_3d_positions (i.e. microns).
        0 corresponds to no shift.
    resolution : np.array of shape (3,)
        Resolution in pixels/microns for each of the three spatial axes.
    uv_mask : None or np.array of shape (uv_grid_steps, uv_grid_steps) of dtype bool
        Mask of covered part of the UV square. If provided, interpolation results are set to np.nan outside the
        covered region. If None, no masking takes place. No masking may result in spurious results
        in the part of the UV square not covered by the unwrapped mesh.
    
    Returns
    -------
    interpolated_data : np.array of shape (n_channels, n_layers, uv_grid_steps, uv_grid_steps)
        3d volumetric data multulayer-interpolated onto UV grid.
    
    """
    interpolated_normals = (interpolated_normals.T / np.linalg.norm(interpolated_normals, axis=-1).T).T
    interpolated_data = np.stack([interpolate_volumetric_data_to_uv(image,
                                  interpolated_3d_positions+o*interpolated_normals, resolution, uv_mask=uv_mask)
                                  for o in normal_offsets], axis=1)
    return interpolated_data

# %% ../nbs/02_cartographic_interpolation.ipynb 43
def create_cartographic_projections(image, mesh, resolution, normal_offsets=(0,), uv_grid_steps=256,
                                    uv_mask='auto'):
    """
    Create multilayer cartographic projections of image using mesh.
    
    Computes multiple layers along surface normal, with given normal offset (in microns). 0 offset
    corresponds to no shift away from the mesh. Also computes 3d positions (in microns)
    and surface normals interpolated onto the UV grid.
    
    UV positions that don't correspond to any 3d position are set to np.nan.
    
    Parameters
    ----------
    image : str or 4d np.array
        Image, either as path to file, or as array. If array, axis 0  is assumed to be the channel axis
    mesh : str or dict
        Mesh, either as path to file, or as dict with entries "vertices", "texture_vertices", "normals",
        and "faces"
    resolution : np.array of shape (3,)
        Image resolution in pixels/micron for the three spatial axes
    normal_offsets : np.array of float, optional
        Offsets along normal direction, in same units as interpolated_3d_positions (i.e. microns).
        0 corresponds to no shift.
    uv_grid_steps : int, default 256
        Size of UV grid. Determines resolution of result.
    uv_mask : str, None, or np.array of shape (uv_grid_steps, uv_grid_steps) and dtype bool
        Mask of covered part of the UV square. Interpolation results are set to np.nan outside the
        covered region. If "auto", mask is infered automatically from the mesh information.
        If None, no masking takes place. No masking may result in spurious results in the part of
        the UV square not covered by the unwrapped mesh. Can be obtained manually from exported
        blender UV layout.
    
    Returns
    -------
    interpolated_data : np.array of shape (n_channels, n_layers, uv_grid_steps, uv_grid_steps)
        3d volumetric data multulayer-interpolated across [0,1]^2 UV grid, with uniform step size.
    interpolated_3d_positions : np.array of shape (uv_grid_steps, uv_grid_steps, 3)
        3d positions across [0,1]^2 UV grid, with uniform step size. 
    interpolated_normals : np.array of shape (uv_grid_steps, uv_grid_steps, 3)
        Normals across [0,1]^2 UV grid, with uniform step size.
    """
    if isinstance(image, str):
        image = tcio.adjust_axis_order(tcio.imread(image))
    if isinstance(mesh, str):
        mesh = tcio.read_obj(mesh)
    if uv_mask == "auto":
        uv_mask = get_uv_layout_mask_mask(mesh, uv_grid_step=uv_grid_steps)
    matched_mesh_data =  tcio.match_vertex_info(**mesh)
    u, v = 2*[np.linspace(0,1, uv_grid_steps),]
    U, V = np.meshgrid(u, v)
    interpolated_3d_positions = interpolate_3d_to_uv(matched_mesh_data["texture_vertices"],
                                                     matched_mesh_data["vertices"],
                                                     uv_grid_steps=uv_grid_steps, uv_mask=uv_mask)
    interpolated_normals = interpolate_3d_to_uv(matched_mesh_data["texture_vertices"],
                                                matched_mesh_data["normals"],
                                                uv_grid_steps=uv_grid_steps, uv_mask=uv_mask)
    interpolated_data = interpolate_volumetric_data_to_uv_multilayer(image,
                                                                     interpolated_3d_positions,
                                                                     interpolated_normals,
                                                                     normal_offsets,
                                                                     resolution,
                                                                     uv_mask=uv_mask)
    return interpolated_data, interpolated_3d_positions, interpolated_normals 