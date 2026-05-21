"""In-memory conversion between Blender mesh objects and pymeshlab MeshSet."""
import numpy as np
import pymeshlab
import bpy


def blender_to_pymeshlab(obj):
    """
    Convert a Blender mesh object to a pymeshlab MeshSet with one mesh.

    For regular meshes: extracts vertices and triangulated faces.
    For point-cloud-like objects (no faces): extracts only vertices (and normals if available).

    Parameters
    ----------
    obj : bpy.types.Object
        A Blender object with obj.type == 'MESH'.

    Returns
    -------
    pymeshlab.MeshSet
        MeshSet containing one mesh.
    """
    mesh = obj.data
    mesh.calc_loop_triangles()
    mesh.update()

    verts = np.zeros(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get('co', verts)
    verts = verts.reshape(-1, 3)

    if len(mesh.loop_triangles) == 0:
        normals_flat = np.zeros(len(mesh.vertices) * 3, dtype=np.float64)
        mesh.vertices.foreach_get('normal', normals_flat)
        normals = normals_flat.reshape(-1, 3)
        pm_mesh = pymeshlab.Mesh(vertex_matrix=verts, v_normals_matrix=normals)
    else:
        tris = np.zeros(len(mesh.loop_triangles) * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get('vertices', tris)
        tris = tris.reshape(-1, 3)
        pm_mesh = pymeshlab.Mesh(vertex_matrix=verts, face_matrix=tris)

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pm_mesh)
    return ms


def pymeshlab_to_blender(ms, name, context):
    """
    Convert the current mesh of a pymeshlab MeshSet to a new Blender object.

    Parameters
    ----------
    ms : pymeshlab.MeshSet
    name : str
        Name for the new Blender object.
    context : bpy.types.Context

    Returns
    -------
    bpy.types.Object
        Newly created and linked Blender object.
    """
    pm_mesh = ms.current_mesh()
    verts = pm_mesh.vertex_matrix()
    faces = pm_mesh.face_matrix()

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts.tolist(), [], faces.tolist())
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    context.collection.objects.link(obj)
    return obj
