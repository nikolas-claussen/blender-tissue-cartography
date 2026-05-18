"""
Blender mesh utilities: creation, modifiers, numpy attribute storage, and selection helpers.
"""

import bpy
import numpy as np


def create_mesh_from_numpy(name, verts, faces):
    """
    Create a Blender mesh object from NumPy arrays of vertices and faces.

    Parameters
    ----------
    name : str
        Name of the new mesh object.
    verts : np.array of shape (n, 3)
        Vertex coordinates.
    faces : np.array of shape (m, 3 or 4)
        Face vertex indices.

    Returns
    -------
    bpy.types.Object
        The created mesh object.
    """
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    mesh.from_pydata(verts.tolist(), [], faces.tolist())
    mesh.update()
    return obj


def shrinkwrap_and_smooth(source_obj, target_obj, corrective_smooth_iter=0):
    """
    Apply a shrinkwrap modifier to source_obj targeting target_obj, with optional
    corrective smooth iterations, then apply all modifiers.

    Parameters
    ----------
    source_obj : bpy.types.Object
        Source mesh object to be modified.
    target_obj : bpy.types.Object
        Target mesh object for the shrinkwrap modifier.
    corrective_smooth_iter : int, optional
        Number of corrective smooth + shrinkwrap pairs to add. 0 means none.

    Returns
    -------
    bpy.types.Object
        The modified source object.
    """
    if source_obj.type != 'MESH' or target_obj.type != 'MESH':
        raise ValueError("Both source_obj and target_obj must be mesh objects.")

    original_active_obj = bpy.context.view_layer.objects.active

    shrinkwrap_1 = source_obj.modifiers.new(name="Shrinkwrap", type='SHRINKWRAP')
    shrinkwrap_1.target = target_obj
    shrinkwrap_1.wrap_method = 'TARGET_PROJECT'

    for i in range(corrective_smooth_iter):
        corrective_smooth = source_obj.modifiers.new(name=f"Corrective Smooth {i}",
                                                     type='CORRECTIVE_SMOOTH')
        corrective_smooth.iterations = 5
        corrective_smooth.scale = 0
        shrinkwrap_i = source_obj.modifiers.new(name=f"Shrinkwrap {i}", type='SHRINKWRAP')
        shrinkwrap_i.target = target_obj
        shrinkwrap_i.wrap_method = 'TARGET_PROJECT'

    bpy.context.view_layer.objects.active = source_obj
    for modifier in source_obj.modifiers:
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    bpy.context.view_layer.objects.active = original_active_obj
    return source_obj


def set_numpy_attribute(mesh, name, array):
    """
    Store a numpy array as a custom property on a Blender mesh/object.

    Since Blender does not support arbitrary Python objects as properties, the array
    is flattened to bytes and stored together with its shape and dtype as a tuple.
    """
    mesh[name] = (array.flatten().tobytes(), array.shape, array.dtype.str)


def get_numpy_attribute(mesh, name):
    """
    Retrieve a numpy array stored as a custom property on a Blender mesh/object.

    Raises
    ------
    KeyError
        If the attribute is not found.
    """
    if name not in mesh:
        raise KeyError(f"Attribute '{name}' not found on mesh/object")
    return np.frombuffer(mesh[name][0], dtype=mesh[name][2]).reshape(mesh[name][1])


def separate_selected_into_mesh_and_box(self, context):
    """
    From the current selection, separate exactly one mesh and one bounding-box object
    (identified by the '3D_data' custom property).

    Reports an error and returns (None, None) if the selection does not match exactly.
    """
    n_data_selected = sum(1 for x in context.selected_objects if "3D_data" in x)
    n_mesh_selected = sum(1 for x in context.selected_objects if "3D_data" not in x)
    if not (n_data_selected == 1 and n_mesh_selected == 1):
        self.report({'ERROR'}, "Select exactly one mesh and one 3D image (BoundingBox)!")
        return None, None
    box = next(x for x in context.selected_objects if "3D_data" in x)
    obj = next(x for x in context.selected_objects if "3D_data" not in x)
    if obj.type != 'MESH':
        self.report({'ERROR'}, "No mesh object selected!")
        return None, None
    return box, obj
