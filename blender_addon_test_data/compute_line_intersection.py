import bpy
import numpy as np
from mathutils import Vector
from bpy.app.handlers import persistent

# ============================================================
# USER SETTINGS
# ============================================================

MESH_A_NAME = "Mesh_A"
MESH_B_NAME = "Mesh_B"

LINE_OBJECT_NAME = "A_B_intersection_line"

LINE_COLOR = (1.0, 0.05, 0.05, 1.0)  # bright red
LINE_RADIUS = 1 # 0.1
OFFSET_ABOVE_B = 0.002

SNAP_EPS = 1e-5  # removes "spotty" artifacts

# ============================================================
# INTERNAL STATE
# ============================================================

_is_updating = False
_last_matrix_a = None
_last_matrix_b = None

# ============================================================
# MATERIAL
# ============================================================

def get_line_material():
    name = "Intersection_Line_Red"

    mat = bpy.data.materials.get(name)
    if mat:
        return mat

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True

    bsdf = mat.node_tree.nodes.get("Principled BSDF")

    if bsdf:
        bsdf.inputs["Base Color"].default_value = LINE_COLOR
        bsdf.inputs["Emission Color"].default_value = LINE_COLOR
        bsdf.inputs["Emission Strength"].default_value = 2.0

    return mat

# ============================================================
# CURVE OBJECT
# ============================================================

def get_curve_object():
    obj = bpy.data.objects.get(LINE_OBJECT_NAME)

    if obj:
        return obj

    curve = bpy.data.curves.new(LINE_OBJECT_NAME, type="CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = LINE_RADIUS
    curve.bevel_resolution = 6
    curve.fill_mode = 'FULL'

    obj = bpy.data.objects.new(LINE_OBJECT_NAME, curve)
    bpy.context.collection.objects.link(obj)

    obj.show_in_front = True

    mat = get_line_material()
    obj.data.materials.append(mat)

    return obj

def clear_curve(curve):
    while curve.splines:
        curve.splines.remove(curve.splines[0])

# ============================================================
# FAST INTERSECTION (VECTOR + SNAP)
# ============================================================

def snap(p):
    return np.round(p / SNAP_EPS) * SNAP_EPS

def compute_segments(obj_a, obj_b, depsgraph):
    eval_a = obj_a.evaluated_get(depsgraph)
    mesh = eval_a.to_mesh()

    # --- vertices → numpy ---
    coords = np.array([v.co[:] for v in mesh.vertices])

    # --- world transform ---
    Mw = np.array(obj_a.matrix_world)
    coords = coords @ Mw[:3,:3].T + Mw[:3,3]

    # --- into B local space ---
    Minv = np.array(obj_b.matrix_world.inverted())
    coords = coords @ Minv[:3,:3].T + Minv[:3,3]

    z = coords[:,2]

    segments = []

    b_to_world = obj_b.matrix_world
    normal = (obj_b.matrix_world.to_3x3() @ Vector((0,0,1))).normalized()

    for poly in mesh.polygons:
        ids = list(poly.vertices)
        zvals = z[ids]

        # ✅ early reject
        if (zvals > 0).all() or (zvals < 0).all():
            continue

        crossings = []

        for i in range(len(ids)):
            i0 = ids[i]
            i1 = ids[(i+1)%len(ids)]

            z0 = z[i0]
            z1 = z[i1]

            if z0 * z1 > 0:
                continue

            if abs(z0 - z1) < 1e-12:
                continue

            p0 = coords[i0]
            p1 = coords[i1]

            t = -z0 / (z1 - z0)
            p = p0 + t * (p1 - p0)

            crossings.append(snap(p))

        if len(crossings) == 2:
            v0 = b_to_world @ Vector(crossings[0])
            v1 = b_to_world @ Vector(crossings[1])

            v0 += normal * OFFSET_ABOVE_B
            v1 += normal * OFFSET_ABOVE_B

            if (v1 - v0).length > 1e-8:
                segments.append((v0, v1))

    eval_a.to_mesh_clear()
    return segments

# ============================================================
# UPDATE FUNCTION
# ============================================================

def update_intersection():
    global _is_updating, _last_matrix_a, _last_matrix_b

    if _is_updating:
        return

    obj_a = bpy.data.objects.get(MESH_A_NAME)
    obj_b = bpy.data.objects.get(MESH_B_NAME)

    if obj_a is None or obj_b is None:
        return

    # ✅ SKIP if nothing moved
    if (_last_matrix_a is not None and
        _last_matrix_b is not None and
        obj_a.matrix_world == _last_matrix_a and
        obj_b.matrix_world == _last_matrix_b):
        return

    _last_matrix_a = obj_a.matrix_world.copy()
    _last_matrix_b = obj_b.matrix_world.copy()

    _is_updating = True

    try:
        depsgraph = bpy.context.evaluated_depsgraph_get()

        segments = compute_segments(obj_a, obj_b, depsgraph)

        curve_obj = get_curve_object()
        curve = curve_obj.data

        clear_curve(curve)

        for p0, p1 in segments:
            spline = curve.splines.new('POLY')
            spline.points.add(1)

            spline.points[0].co = (*p0, 1)
            spline.points[1].co = (*p1, 1)

    finally:
        _is_updating = False

# ============================================================
# HANDLER
# ============================================================

@persistent
def handler(scene, depsgraph):
    update_intersection()

# ============================================================
# ENABLE / DISABLE
# ============================================================

def enable():
    disable()  # prevent duplicates

    bpy.app.handlers.depsgraph_update_post.append(handler)
    update_intersection()

def disable():
    bpy.app.handlers.depsgraph_update_post[:] = [
        h for h in bpy.app.handlers.depsgraph_update_post
        if h.__name__ != "handler"
    ]

# ============================================================
# RUN
# ============================================================

enable()

#disable() # uncomment this to turn off interactive mode