"""Blender operators for PyMeshLab remeshing operations."""

import bpy
from bpy.types import Operator

from .mesh_utils import blender_to_pymeshlab, pymeshlab_to_blender


def _run_pymeshlab(self, context, apply_fn, suffix):
    """
    Export active object to pymeshlab, apply filter, import result as new object.

    Parameters
    ----------
    self : Operator
        The calling Blender operator (used for self.report).
    context : bpy.types.Context
    apply_fn : callable
        Function(ms: pymeshlab.MeshSet) -> None. Applies filters in-place.
    suffix : str
        Suffix appended to active object name for the new object.

    Returns
    -------
    set
        {'FINISHED'} or {'CANCELLED'}.
    """
    try:
        import pymeshlab  # noqa: F401
    except ImportError as e:
        self.report({'ERROR'}, f"PyMeshLab failed to load: {e}")
        return {'CANCELLED'}

    if context.active_object is None or context.active_object.type != 'MESH':
        self.report({'ERROR'}, "Select a mesh object first.")
        return {'CANCELLED'}

    obj = context.active_object
    ms = blender_to_pymeshlab(obj)

    try:
        apply_fn(ms)
    except Exception as e:
        self.report({'ERROR'}, str(e))
        return {'CANCELLED'}

    if ms.current_mesh().vertex_number() == 0:
        self.report({'WARNING'}, "Result mesh has no vertices; check input and parameters.")
        return {'CANCELLED'}

    new_name = obj.name + suffix
    pymeshlab_to_blender(ms, new_name, context)
    self.report({'INFO'}, f"Created '{new_name}'.")
    return {'FINISHED'}


class IsotropicRemeshOperator(Operator):
    """Apply isotropic explicit remeshing via PyMeshLab."""

    bl_idname = "scene.pymeshlab_iso_remesh"
    bl_label = "Apply Isotropic Remeshing"
    bl_description = "Remesh with uniform triangle size. Ideal after marching cubes. Erases existing UV."

    def execute(self, context):
        import pymeshlab  # needed by apply_fn closure (pymeshlab.PercentageValue)
        scene = context.scene
        iterations = scene.pymeshlab_iso_iterations
        targetlen = scene.pymeshlab_iso_targetlen
        featuredeg = scene.pymeshlab_iso_featuredeg

        def apply_fn(ms):
            ms.meshing_isotropic_explicit_remeshing(
                iterations=iterations,
                targetlen=pymeshlab.PercentageValue(targetlen),
                featuredeg=featuredeg,
            )

        return _run_pymeshlab(self, context, apply_fn, "_iso_remesh")


class QuadricDecimateOperator(Operator):
    """Apply quadric edge collapse decimation via PyMeshLab."""

    bl_idname = "scene.pymeshlab_decimate"
    bl_label = "Apply Quadric Decimation"
    bl_description = "Reduce face count while preserving shape via quadric edge collapse."

    def execute(self, context):
        scene = context.scene
        mode = scene.pymeshlab_decim_mode
        facenum = scene.pymeshlab_decim_facenum
        perc = scene.pymeshlab_decim_perc

        def apply_fn(ms):
            if mode == 'FACE_COUNT':
                ms.meshing_decimation_quadric_edge_collapse(targetfacenum=facenum)
            else:
                ms.meshing_decimation_quadric_edge_collapse(targetperc=perc)

        return _run_pymeshlab(self, context, apply_fn, "_decimated")


class PoissonReconstructOperator(Operator):
    """Apply screened Poisson surface reconstruction via PyMeshLab."""

    bl_idname = "scene.pymeshlab_poisson"
    bl_label = "Apply Poisson Reconstruction"
    bl_description = "Reconstruct a surface from a point cloud using screened Poisson reconstruction."

    def execute(self, context):
        scene = context.scene
        compute_normals = scene.pymeshlab_poisson_normals
        k = scene.pymeshlab_poisson_k
        depth = scene.pymeshlab_poisson_depth
        fulldepth = scene.pymeshlab_poisson_fulldepth

        def apply_fn(ms):
            if compute_normals:
                ms.compute_normal_for_point_clouds(k=k, smoothiter=2)
            ms.generate_surface_reconstruction_screened_poisson(
                depth=depth, fulldepth=fulldepth
            )

        return _run_pymeshlab(self, context, apply_fn, "_poisson")


class AlphaWrapOperator(Operator):
    """Apply alpha wrap via PyMeshLab."""

    bl_idname = "scene.pymeshlab_alphawrap"
    bl_label = "Apply Alpha Wrap"
    bl_description = "Generate a watertight surface from any mesh, including non-manifold and open boundaries."

    def execute(self, context):
        scene = context.scene
        alpha = scene.pymeshlab_alphawrap_alpha
        offset = scene.pymeshlab_alphawrap_offset

        def apply_fn(ms):
            ms.generate_alpha_wrap(alpha_fraction=alpha, offset_fraction=offset)

        return _run_pymeshlab(self, context, apply_fn, "_alphawrap")


class MeshCleanupOperator(Operator):
    """Apply mesh cleanup operations via PyMeshLab."""

    bl_idname = "scene.pymeshlab_cleanup"
    bl_label = "Apply Mesh Cleanup"
    bl_description = "Remove duplicate vertices, null faces, unreferenced vertices, and non-manifold edges."

    def execute(self, context):
        scene = context.scene
        close_holes = scene.pymeshlab_cleanup_close_holes
        maxholesize = scene.pymeshlab_cleanup_maxholesize

        def apply_fn(ms):
            ms.meshing_remove_duplicate_vertices()
            ms.meshing_remove_null_faces()
            ms.meshing_remove_unreferenced_vertices()
            ms.meshing_repair_non_manifold_edges()
            if close_holes:
                ms.meshing_close_holes(maxholesize=maxholesize)

        return _run_pymeshlab(self, context, apply_fn, "_clean")


OPERATOR_CLASSES = [
    IsotropicRemeshOperator,
    QuadricDecimateOperator,
    PoissonReconstructOperator,
    AlphaWrapOperator,
    MeshCleanupOperator,
]
