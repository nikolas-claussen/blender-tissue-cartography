# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/01c_interface_open3d.ipynb.

# %% auto 0
__all__ = ['convert_to_open3d', 'convert_to_pymeshlab', 'convert_from_pymeshlab']

# %% ../nbs/01c_interface_open3d.ipynb 1
from . import io as tcio
import numpy as np
import open3d as o3d

# %% ../nbs/01c_interface_open3d.ipynb 5
def convert_to_open3d(mesh: tcio.ObjMesh, recompute_normals=True,
                      add_texture_info=None) -> o3d.t.geometry.TriangleMesh:
    """
    Convert tcio.ObjMesh to open3d.t.geometry.TriangleMesh.
    
    See https://www.open3d.org/docs/latest/python_api/open3d.t.geometry.TriangleMesh.html
    Note: open3d stores its texture coordinates generally as face attributes.
    The returned mesh has both a vertex attribute mesh_o3d.vertex.texture_uvs
    and a face attribute mesh_o3d.triangle.texture_uvs for compatibility with the
    open3d UV algorithms.
    
    Parameters
    ----------
    mesh : tcio.ObjMesh
    recompute_normals : bool, default True
    add_texture_info : None or bool
        Whether to add texture info to the pymeshlab.Mesh. If None, texture is added if available
        for all vertices.
    Returns
    -------
    mesh_o3d: o3d.t.geometry.TriangleMesh

    """

    mesh.match_vertex_info(require_texture_normals=False)
    add_texture_info = (not np.isnan(mesh.matched_texture_vertices).any()
                    if add_texture_info is None else add_texture_info)
    
    dtype_f = o3d.core.float32
    dtype_i = o3d.core.int32
    mesh_o3d = o3d.t.geometry.TriangleMesh()
    
    mesh_o3d.triangle.indices = o3d.core.Tensor(mesh.tris, dtype_i)
    mesh_o3d.vertex.positions = o3d.core.Tensor(mesh.matched_vertices, dtype_f)
    if recompute_normals:
        mesh_o3d.compute_vertex_normals()
    else:
        mesh_o3d.vertex.normals = o3d.core.Tensor(mesh.matched_normals, dtype_f)
    if add_texture_info:
        mesh_o3d.vertex.texture_uvs = o3d.core.Tensor(mesh.matched_texture_vertices, dtype_f)
        mesh_o3d.triangle.texture_uvs =o3d.core.Tensor(np.stack([[mesh.matched_texture_vertices[v] for v in tri]
                                                                 for tri in mesh.tris]), dtype_f)
    return mesh_o3d

# %% ../nbs/01c_interface_open3d.ipynb 31
def convert_to_pymeshlab(mesh: tcio.ObjMesh) -> pymeshlab.Mesh:
    """
    Convert tcio.ObjMesh to pymeshlab.Mesh.
    
    See https://pymeshlab.readthedocs.io/en/latest/classes/mesh.html
    Note: normal information is recalculated by pymeshlab. Discards any non-triangle faces.
    """
    mesh.match_vertex_info(require_texture_normals=False)
    if np.isnan(mesh.matched_texture_vertices).any():
        return pymeshlab.Mesh(vertex_matrix=mesh.matched_vertices, face_matrix=mesh.tris)
    return pymeshlab.Mesh(vertex_matrix=mesh.matched_vertices, face_matrix=mesh.tris,
                          v_tex_coords_matrix=mesh.matched_texture_vertices.astype(np.float64))

# %% ../nbs/01c_interface_open3d.ipynb 37
def convert_from_pymeshlab(mesh: pymeshlab.pmeshlab.Mesh) -> pymeshlab.Mesh:
    """Convert pymeshlab mesh to ObjMesh."""
    vertices = mesh.vertex_matrix()
    faces = [[3*[v,] for v in f] for f in pymesh_ref.face_matrix()]
    normals = pymesh_ref.vertex_normal_matrix()
    normals = (normals.T / np.linalg.norm(normals, axis=-1)).T
    if mesh.has_vertex_tex_coord():
        return tcio.ObjMesh(vertices=vertices, faces=faces, normals=normals,
                            texture_vertices=mesh.vertex_tex_coord_matrix())
    return tcio.ObjMesh(vertices=vertices, faces=faces, normals=normals)