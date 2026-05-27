"""PyMeshLab Remeshing — Blender extension entry point."""

import bpy
from bpy.props import (
    FloatProperty, IntProperty, BoolProperty, EnumProperty,
)

from .operators import OPERATOR_CLASSES
from .ux import MeshLabRemeshPanel


def register():
    """Register all add-on classes and scene properties."""
    for cls in OPERATOR_CLASSES:
        bpy.utils.register_class(cls)
    bpy.utils.register_class(MeshLabRemeshPanel)

    # --- Isotropic Remeshing ---
    bpy.types.Scene.pymeshlab_iso_iterations = IntProperty(
        name="Iterations", default=10, min=1,
    )
    bpy.types.Scene.pymeshlab_iso_targetlen = FloatProperty(
        name="Target Edge Length (%)",
        default=1.0,
        description="Target edge length as % of bbox diagonal",
    )
    bpy.types.Scene.pymeshlab_iso_featuredeg = FloatProperty(
        name="Feature Angle",
        default=30.0,
        description="Feature angle threshold (deg) — edges sharper than this are preserved",
    )

    # --- Quadric Decimation ---
    bpy.types.Scene.pymeshlab_decim_mode = EnumProperty(
        name="Mode",
        items=[
            ('FACE_COUNT', 'Face Count', 'Target a specific face count'),
            ('PERCENTAGE', 'Percentage', 'Target a fraction of input faces'),
        ],
        default='FACE_COUNT',
    )
    bpy.types.Scene.pymeshlab_decim_facenum = IntProperty(
        name="Face Count", default=1000, min=1,
    )
    bpy.types.Scene.pymeshlab_decim_perc = FloatProperty(
        name="Fraction",
        default=0.5, min=0.01, max=0.99,
        description="Target fraction of original faces (0–1)",
    )

    # --- Screened Poisson Reconstruction ---
    bpy.types.Scene.pymeshlab_poisson_normals = BoolProperty(
        name="Compute Normals",
        default=True,
        description="Compute/smooth normals before reconstruction. Required for point clouds without normals.",
    )
    bpy.types.Scene.pymeshlab_poisson_k = IntProperty(
        name="Normal Neighbors",
        default=10, min=3,
        description="Neighbors used for normal estimation",
    )
    bpy.types.Scene.pymeshlab_poisson_depth = IntProperty(
        name="Reconstruction Depth",
        default=8, min=1, max=14,
        description="Reconstruction depth (higher = finer, slower)",
    )
    bpy.types.Scene.pymeshlab_poisson_fulldepth = IntProperty(
        name="Adaptive Depth",
        default=5, min=1, max=14,
        description="Adaptive octree depth",
    )

    # --- Alpha Wrap ---
    bpy.types.Scene.pymeshlab_alphawrap_alpha = FloatProperty(
        name="Alpha",
        default=0.02, min=0.001, max=0.5,
        description="Ball size as fraction of bbox diagonal — smaller = more detail",
    )
    bpy.types.Scene.pymeshlab_alphawrap_offset = FloatProperty(
        name="Offset",
        default=0.001, min=0.0001, max=0.1,
        precision=4,
        description="Surface offset as fraction of bbox diagonal",
    )

    # --- Mesh Cleanup ---
    bpy.types.Scene.pymeshlab_cleanup_close_holes = BoolProperty(
        name="Close Holes",
        default=False,
        description="Fill holes after cleanup",
    )
    bpy.types.Scene.pymeshlab_cleanup_maxholesize = IntProperty(
        name="Max Hole Size",
        default=30, min=1,
        description="Max hole perimeter (number of edges) to fill",
    )


def unregister():
    """Unregister all add-on classes and scene properties.

    Guards against double-unregister that occurs when a hot-reload script
    reloads modules before calling addon_disable: the reloaded class objects
    are new Python objects without bl_rna, so unregister_class would raise.
    """
    for cls in reversed(OPERATOR_CLASSES):
        if hasattr(cls, 'bl_rna'):
            bpy.utils.unregister_class(cls)
    if hasattr(MeshLabRemeshPanel, 'bl_rna'):
        bpy.utils.unregister_class(MeshLabRemeshPanel)

    props = [
        'pymeshlab_iso_iterations', 'pymeshlab_iso_targetlen', 'pymeshlab_iso_featuredeg',
        'pymeshlab_decim_mode', 'pymeshlab_decim_facenum', 'pymeshlab_decim_perc',
        'pymeshlab_poisson_normals', 'pymeshlab_poisson_k',
        'pymeshlab_poisson_depth', 'pymeshlab_poisson_fulldepth',
        'pymeshlab_alphawrap_alpha', 'pymeshlab_alphawrap_offset',
        'pymeshlab_cleanup_close_holes', 'pymeshlab_cleanup_maxholesize',
    ]
    for prop in props:
        if hasattr(bpy.types.Scene, prop):
            delattr(bpy.types.Scene, prop)


if __name__ == "__main__":
    register()
