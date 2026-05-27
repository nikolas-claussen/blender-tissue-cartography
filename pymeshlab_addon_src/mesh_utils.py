"""In-memory conversion between Blender mesh objects and pymeshlab MeshSet."""
import numpy as np
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
    import pymeshlab
    mesh = obj.data
    mesh.update()
    mesh.calc_loop_triangles()

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
    import pymeshlab  # noqa: F401 — ensures numpy is loaded via pymeshlab's context
    pm_mesh = ms.current_mesh()
    verts = pm_mesh.vertex_matrix()
    faces = pm_mesh.face_matrix()
    n_verts, n_faces = len(verts), len(faces)

    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(n_verts)
    mesh.vertices.foreach_set('co', verts.ravel())
    mesh.loops.add(n_faces * 3)
    mesh.loops.foreach_set('vertex_index', faces.ravel())
    mesh.polygons.add(n_faces)
    mesh.polygons.foreach_set('loop_start', np.arange(0, n_faces * 3, 3, dtype=np.int32))
    mesh.polygons.foreach_set('loop_total', np.full(n_faces, 3, dtype=np.int32))
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    context.collection.objects.link(obj)
    return obj
