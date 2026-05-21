"""UI panel for PyMeshLab Remeshing."""

import bpy
from bpy.types import Panel


class MeshLabRemeshPanel(Panel):
    """Panel for PyMeshLab remeshing operations."""

    bl_label = "PyMeshLab Remeshing"
    bl_idname = "SCENE_PT_pymeshlab_remesh"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "scene"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        layout.label(text="PyMeshLab Remeshing", icon='MESH_DATA')
        layout.label(text="Select a mesh or point cloud object in the scene, then apply an operation below.")

        # --- Isotropic Remeshing ---
        box = layout.box()
        box.label(text="Isotropic Remeshing", icon='MOD_REMESH')
        box.label(text="Uniform triangle size. Ideal after marching cubes. Erases UV.")
        box.prop(scene, "pymeshlab_iso_iterations")
        box.prop(scene, "pymeshlab_iso_targetlen")
        box.prop(scene, "pymeshlab_iso_featuredeg")
        box.operator("scene.pymeshlab_iso_remesh")

        # --- Quadric Decimation ---
        box = layout.box()
        box.label(text="Quadric Decimation", icon='MOD_DECIM')
        box.label(text="Reduce face count while preserving shape.")
        box.prop(scene, "pymeshlab_decim_mode")
        if scene.pymeshlab_decim_mode == 'FACE_COUNT':
            box.prop(scene, "pymeshlab_decim_facenum")
        else:
            box.prop(scene, "pymeshlab_decim_perc")
        box.operator("scene.pymeshlab_decimate")

        # --- Screened Poisson Reconstruction ---
        box = layout.box()
        box.label(text="Screened Poisson Reconstruction", icon='POINTCLOUD_DATA')
        box.label(text="Surface from point cloud. Input must be a mesh or point cloud with vertex positions.")
        box.prop(scene, "pymeshlab_poisson_normals")
        if scene.pymeshlab_poisson_normals:
            row = box.row()
            row.separator()
            row.prop(scene, "pymeshlab_poisson_k")
        box.prop(scene, "pymeshlab_poisson_depth")
        box.prop(scene, "pymeshlab_poisson_fulldepth")
        box.operator("scene.pymeshlab_poisson")

        # --- Alpha Wrap ---
        box = layout.box()
        box.label(text="Alpha Wrap", icon='MESH_ICOSPHERE')
        box.label(text="Watertight surface from any mesh, including non-manifold and open boundaries.")
        box.prop(scene, "pymeshlab_alphawrap_alpha")
        box.prop(scene, "pymeshlab_alphawrap_offset")
        box.operator("scene.pymeshlab_alphawrap")

        # --- Mesh Cleanup ---
        box = layout.box()
        box.label(text="Mesh Cleanup", icon='BRUSH_DATA')
        box.label(text="Remove duplicates, null faces, unreferenced vertices, non-manifold edges.")
        box.prop(scene, "pymeshlab_cleanup_close_holes")
        if scene.pymeshlab_cleanup_close_holes:
            box.prop(scene, "pymeshlab_cleanup_maxholesize")
        box.operator("scene.pymeshlab_cleanup")
