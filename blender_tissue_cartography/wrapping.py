# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/Python library/05c_wrapping.ipynb.

# %% auto 0
__all__ = ['shrinkwrap_igl']

# %% ../nbs/Python library/05c_wrapping.ipynb 3
from . import io as tcio
from . import mesh as tcmesh
from . import registration as tcreg
from . import smoothing as tcsmooth

import numpy as np
from copy import deepcopy
import warnings
import igl

from scipy import sparse

# %% ../nbs/Python library/05c_wrapping.ipynb 21
def shrinkwrap_igl(mesh_source, mesh_target, n_iter_smooth_target=10, n_iter_smooth_wrapped=10):
    """
    Shrink-wrap the source mesh onto the target mesh using trimesh.
    
    Sets the vertex positions of mesh_source to the closest point on the surface of mesh_target (not necessarily
    a vertex). Optionally, smooth the target mesh and the wrapped mesh for smoother results using a Taubin
    filter (recommended). Gives out a warning if the shrink-wrapping flips any vertex normals, which can
    indicate problems.
    
    The shrinkwrapped mesh still has the UV maps of the source mesh, and so can be used to compute
    cartographic projections. Assumes mesh is triangular.
    
    Parameters
    ----------
    mesh_source : tcmesh.ObjMesh
        Mesh to be deformed
    mesh_target : tcmesh.ObjMesh
        Mesh with the target shape
    n_iter_smooth_target : int, default 10
        Taubin smoothing iterations for target
    n_iter_smooth_wrapped : int, default 10
        Taubin smoothing iterations for shrinkwrapped mesh, after shrinkwrapping

    Returns
    -------
    mesh_wrapped : tcmesh.ObjMesh

    """
    if not mesh_target.is_triangular:
        warnings.warn(f"Warning: mesh not triangular - result may be incorrect", RuntimeWarning)
    # smooth if necessary
    if n_iter_smooth_target > 0:
        target_verts = tcsmooth.smooth_taubin(mesh_target, n_iter=n_iter_smooth_target).vertices
    else:
        target_verts = mesh_target.vertices
    # compute closest point on target mesh for each source vertex
    distances, indices, points = igl.point_mesh_squared_distance(mesh_source.vertices,
                                                                 target_verts, mesh_target.tris)
    # create wrapped mesh
    mesh_wrapped = tcmesh.ObjMesh(points, mesh_source.faces, texture_vertices=mesh_source.texture_vertices,
                                normals=None, name=mesh_source.name)
    mesh_wrapped.set_normals()
    if n_iter_smooth_wrapped > 0:
        mesh_wrapped = tcsmooth.smooth_taubin(mesh_wrapped, n_iter=n_iter_smooth_wrapped)
    # check if any normals were flipped
    dots = np.einsum("vi,vi->v", mesh_source.normals, mesh_wrapped.normals)
    if np.sum(dots < 0) > 0:
        warnings.warn(f"Warning: {np.sum(dots<0)} normal(s) flipped during shrink-wrapping", RuntimeWarning)
    return mesh_wrapped
