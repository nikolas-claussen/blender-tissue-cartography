import bpy
import bmesh
import mathutils
import math
import numpy as np


def map_mesh_to_unit_disk(obj, uv_layer_name="disk_UV"):
    """Map mesh to unit disk. Important - mesh should have a single boundary!"""
    originally_active = bpy.context.view_layer.objects.active
    # Ensure the object is a mesh
    if obj.type != 'MESH':
        raise ValueError("Selected object must be a mesh.")
    # ensure object is active and enter edit mode
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    # clear all pinned vertices
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.pin(clear=True)
    # Add a fresh UV layer
    if uv_layer_name in obj.data.uv_layers:
        obj.data.uv_layers.remove(obj.data.uv_layers[uv_layer_name])
    bpy.ops.mesh.uv_texture_add()
    obj.data.uv_layers[-1].name = uv_layer_name
    obj.data.uv_layers.active = obj.data.uv_layers[uv_layer_name]
    # Perform initial unwrapping
    bpy.ops.uv.unwrap(method='MINIMUM_STRETCH', margin=0.001)
    # Select the boundary loop
    bpy.ops.mesh.select_mode(type='VERT')  # Switch to vertex select mode
    bpy.ops.mesh.select_non_manifold(extend=False)  # Select non-manifold edges

    # Move and pin the boundary vertices
    centroid = [0.5, 0.5]
    bm = bmesh.from_edit_mesh(obj.data)
    for vert in bm.verts:
        if vert.select:
            for loop in vert.link_loops:
                uv = loop[bm.loops.layers.uv.active].uv
                direction = (uv[0] - centroid[0], uv[1] - centroid[1])
                length = math.sqrt(direction[0]**2 + direction[1]**2)
                if length > 0:
                    uv[0] = centroid[0] + 0.5*direction[0] / length
                    uv[1] = centroid[1] + 0.5*direction[1] / length
                loop[bm.loops.layers.uv.active].pin_uv = True
    bmesh.update_edit_mesh(obj.data)
    # Perform second unwrapping
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap(method='MINIMUM_STRETCH', margin=0.001)
    # restore previous select state
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.objects.active = originally_active


# Example usage
    
obj = bpy.context.active_object
map_mesh_to_unit_disk(obj)