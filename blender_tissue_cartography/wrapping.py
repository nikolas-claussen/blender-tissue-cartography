# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/03b_wrapping.ipynb.

# %% auto 0
__all__ = ['get_uniform_laplacian', 'smooth_laplacian', 'smooth_taubin', 'shrinkwrap_igl']

# %% ../nbs/03b_wrapping.ipynb 1
from . import io as tcio
from . import registration as tcreg

import numpy as np
from copy import deepcopy
import warnings
import igl

from scipy import sparse

# %% ../nbs/03b_wrapping.ipynb 17
def get_uniform_laplacian(tris, normalize=True):
    """
    Get uniform Laplacian (purely connectivity based) as sparse matrix.
    
    If normalize, the diagonal = -1. Else, diagonal equals number of neighbors.
    """
    a = igl.adjacency_matrix(tris)
    a_sum = np.squeeze(np.asarray(a.sum(axis=1)))
    a_diag = sparse.diags(a_sum)
    if normalize:
        return sparse.diags(1/a_sum)@(a - a_diag)
    return (a - a_diag)

# %% ../nbs/03b_wrapping.ipynb 18
def smooth_laplacian(mesh: tcio.ObjMesh, lamb=0.5, n_iter=10, method="explicit") -> tcio.ObjMesh:
    """
    Smooth using Laplacian filter.
    
    Assumes mesh is triangular.
    
    Parameters
    ----------
    mesh : ObjMesh
        Initial mesh.
    lamb : float, default 0.5
        Filter strength. Higher = more smoothing.
    n_iter : int
        Filter iterations
    method : str, default "explicit"
        Can use explicit (fast, simple) or implicit (slow, more accurate) method.

    Returns
    -------
    mesh_smoothed : ObjMesh
        Smoothed mesh.

    """
    if not mesh.is_triangular:
        warnings.warn(f"Warning: mesh not triangular - result may be incorrect", RuntimeWarning)
    v_smoothed = np.copy(mesh.vertices)
    f = mesh.tris
    if method == "implicit":
        laplacian = igl.cotmatrix(v, f)
        for _ in range(n_iter):
            mass = igl.massmatrix(v_smoothed, f, igl.MASSMATRIX_TYPE_BARYCENTRIC)
            v_smoothed = sparse.linalg.spsolve(mass - lamb * laplacian, mass.dot(v_smoothed))
    elif method == "explicit":
        laplacian_uniform = get_uniform_laplacian(f)
        for _ in range(n_iter):
            v_smoothed += lamb*laplacian_uniform.dot(v_smoothed)
    mesh_smoothed = tcio.ObjMesh(v_smoothed, mesh.faces, texture_vertices=mesh.texture_vertices,
                                 normals=None, name=mesh.name)
    mesh_smoothed.set_normals()
    return mesh_smoothed

# %% ../nbs/03b_wrapping.ipynb 21
def smooth_taubin(mesh: tcio.ObjMesh, lamb=0.5, nu=0.53, n_iter=10,) -> tcio.ObjMesh:
    """
    Smooth using Taubin filter (like Laplacian, but avoids shrinkage).
    
    Assumes mesh is triangular. See   "Improved Laplacian Smoothing of Noisy Surface Meshes"
    J. Vollmer, R. Mencl, and H. Muller.
    
    Parameters
    ----------
    mesh : ObjMesh
        Initial mesh.
    lamb : float, default 0.5
        Filter strength. Higher = more smoothing.
    nu : float, default 0.53
        Counteract shrinkage. Higher = more dilation.
    n_iter : int
        Filter iterations

    Returns
    -------
    mesh_smoothed : ObjMesh
        Smoothed mesh.

    """
    if not mesh.is_triangular:
        warnings.warn(f"Warning: mesh not triangular - result may be incorrect", RuntimeWarning)
    v_smoothed = np.copy(mesh.vertices)
    laplacian_uniform = get_uniform_laplacian(mesh.tris)
    for _ in range(n_iter):
        v_smoothed += lamb*laplacian_uniform.dot(v_smoothed)
        v_smoothed -= nu*laplacian_uniform.dot(v_smoothed)
    mesh_smoothed = tcio.ObjMesh(v_smoothed, mesh.faces, texture_vertices=mesh.texture_vertices,
                                 normals=None, name=mesh.name)
    mesh_smoothed.set_normals()
    return mesh_smoothed

# %% ../nbs/03b_wrapping.ipynb 31
def shrinkwrap_igl(mesh_source, mesh_target, n_iter_smooth_target=10, n_iter_smooth_wrapped=10):
    """
    Shrink-wrap the source mesh onto the target mesh using trimesh.
    
    Sets the vertex positions of mesh_source to the closes point on the surface of mesh_target (not necessarily
    a vertex). Optionally, smoothes the target mesh and the wrapped mesh for smoother results using a Taubin
    filter (recommended). Gives out a warning if the shrink-wrapping flips any vertex normals, which can
    indicate problems.
    
    The shrinkwrapped mesh still has the UV maps of the source mesh, and so can be used to compute
    cartographic projections. Assumes mesh is triangular.
    
    Parameters
    ----------
    mesh_source : tcio.ObjMesh
        Mesh to be deformed
    mesh_target : tcio.ObjMesh
        Mesh with target shape
    n_iter_smooth_target : int, default 10
        Taubin smoothing iterations for target
    n_iter_smooth_wrapped : int, default 10
        Taubin smoothing iterations for shrinkwrapped mesh, after shrinkwrapping

    Returns
    -------
    mesh_wrapped : tcio.ObjMesh

    """
    if not mesh_target.is_triangular:
        warnings.warn(f"Warning: mesh not triangular - result may be incorrect", RuntimeWarning)
    # smooth if necessary
    if n_iter_smooth_target > 0:
        target_verts = smooth_taubin(mesh_target, n_iter=n_iter_smooth_target).vertices
    else:
        target_verts = mesh_target.vertices
    # compute closest point on target mesh for each source vertex
    distances, indices, points = igl.point_mesh_squared_distance(mesh_source.vertices,
                                                                 target_verts, mesh_target.tris)
    # create wrapped mesh
    mesh_wrapped = tcio.ObjMesh(points, mesh_source.faces, texture_vertices=mesh_source.texture_vertices,
                                normals=None, name=mesh_source.name)
    mesh_wrapped.set_normals()
    if n_iter_smooth_wrapped > 0:
        mesh_wrapped = smooth_taubin(mesh_wrapped, n_iter=n_iter_smooth_wrapped)
    # check if any normals were flipped
    dots = np.einsum("vi,vi->v", mesh_source.normals, mesh_wrapped.normals)
    if np.sum(dots < 0) > 0:
        warnings.warn(f"Warning: {np.sum(dots<0)} normal(s) flipped during shrink-wrapping", RuntimeWarning)
    return mesh_wrapped