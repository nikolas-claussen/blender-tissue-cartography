"""
User interface panel for the Tissue Cartography add-on.
"""

import bpy
from bpy.types import Panel


class TissueCartographyPanel(Panel):
    """Panel defining the Tissue Cartography user interface layout."""
    bl_label = "Tissue Cartography"
    bl_idname = "SCENE_PT_tissue_cartography"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "scene"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # --- 3D image loading ---
        layout.prop(scene, "tissue_cartography_file")
        row_tiff = layout.row()
        row_tiff.prop(scene, "tissue_cartography_resolution")
        row_tiff.prop(scene, "tissue_cartography_axis_order")
        layout.operator("scene.load_tiff", text="Load .tiff file")
        layout.label(
            text=(f"Loaded Image Shape: {scene.tissue_cartography_image_shape}. "
                  f"Loaded Image Channels: {scene.tissue_cartography_image_channels}")
        )
        layout.separator()

        # --- Segmentation loading ---
        layout.prop(scene, "tissue_cartography_segmentation_file")
        row_seg = layout.row()
        row_seg.prop(scene, "tissue_cartography_segmentation_resolution")
        row_seg.prop(scene, "tissue_cartography_segmentation_axis_order")
        row_seg.prop(scene, "tissue_cartography_segmentation_sigma")
        layout.operator("scene.load_segmentation",
                        text="Get mesh(es) from binary segmentation .tiff file(s)")
        layout.label(
            text=(f"Loaded Segmentation Shape: {scene.tissue_cartography_segmentation_shape}. "
                  f"Loaded Segmentation Channels: {scene.tissue_cartography_segmentation_channels}")
        )
        layout.separator()

        # --- Ortho-slice ---
        row_slice = layout.row()
        row_slice.prop(scene, "tissue_cartography_slice_axis")
        row_slice.prop(scene, "tissue_cartography_slice_position")
        row_slice.prop(scene, "tissue_cartography_slice_channel")
        layout.operator("scene.create_slice_plane", text="Create slice plane")
        layout.separator()

        # --- Vertex shader ---
        row_vertex = layout.row()
        row_vertex.prop(scene, "tissue_cartography_vertex_shader_offset")
        row_vertex.prop(scene, "tissue_cartography_vertex_shader_channel_RGB")
        row_vertex2 = layout.row()
        row_vertex2.prop(scene, "tissue_cartography_vertex_shader_create_material")
        row_vertex2.operator("scene.vertex_shader", text="Initialize/refresh vertex shading")
        layout.separator()

        # --- Projection ---
        row_proj = layout.row()
        row_proj.prop(scene, "tissue_cartography_offsets")
        row_proj.prop(scene, "tissue_cartography_projection_resolution")
        row_proj2 = layout.row()
        row_proj2.operator("scene.create_projection", text="Create Projection")
        row_proj2.operator("scene.save_projection", text="Save Projection")
        layout.separator()

        # --- Batch processing ---
        row_batch = layout.row()
        row_batch.prop(scene, "tissue_cartography_batch_directory")
        row_batch.prop(scene, "tissue_cartography_batch_output_directory")
        row_batch.prop(scene, "tissue_cartography_batch_create_materials")
        layout.operator("scene.batch_projection", text="Batch Process And Save")
        layout.separator()

        # --- Alignment ---
        row_align = layout.row()
        row_align.prop(scene, "tissue_cartography_prealign")
        row_align.prop(scene, "tissue_cartography_prealign_shear")
        row_align.prop(scene, "tissue_cartography_prealign_scale")
        row_align.prop(scene, "tissue_cartography_align_type")
        row_align.prop(scene, "tissue_cartography_align_iter")
        layout.operator("scene.align", text="Align Meshes")
        layout.separator()

        # --- Shrinkwrap ---
        row_shrink = layout.row()
        row_shrink.prop(scene, "tissue_cartography_shrinkwrap_smooth")
        row_shrink.prop(scene, "tissue_cartography_shrinkwrap_iterative")
        layout.operator("scene.shrinkwrap", text="Shrinkwrap Meshes (Active To Selected)")
        layout.separator()

        layout.operator("scene.help_popup", text="Show help", icon='HELP')
