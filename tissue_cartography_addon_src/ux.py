"""
User interface panel for the Tissue Cartography add-on.
"""

import bpy
from bpy.types import Panel

from .operators import prune_orphaned_datasets


class TissueCartographyPanel(Panel):
    """Panel defining the Tissue Cartography user interface layout."""
    bl_label = "Tissue Cartography (V2)"
    bl_idname = "SCENE_PT_tissue_cartography"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "scene"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        layout.separator(type='LINE')
        layout.label(text="DATA LOADING", icon='IMPORT')
        # --- 3D image loading ---
        layout.prop(scene, "tissue_cartography_file")
        row_tiff = layout.row()
        row_tiff.prop(scene, "tissue_cartography_resolution")
        row_tiff.prop(scene, "tissue_cartography_axis_order")
        layout.operator("scene.load_tiff", text="Load .tiff file")
        layout.label(
            text=("⚠️ Check that axis order is as expected (or adjust & reload). "
                  f"Loaded Shape: {scene.tissue_cartography_image_shape}. "
                  f"Loaded Channels: {scene.tissue_cartography_image_channels}."
                  )
        )

        # --- Loaded datasets collapsible ---
        datasets_box = layout.box()
        header_row = datasets_box.row()
        header_row.prop(
            scene, "tissue_cartography_show_datasets",
            icon='TRIA_DOWN' if scene.tissue_cartography_show_datasets else 'TRIA_RIGHT',
            icon_only=True, emboss=False,
        )
        prune_orphaned_datasets()  # frees data of deleted boxes; plain dict, safe in draw
        data_store = bpy.types.Scene.tissue_cartography_3D_data
        header_row.label(text=f"Loaded 3D Datasets ({len(data_store)})")
        if scene.tissue_cartography_show_datasets:
            if not data_store:
                datasets_box.label(text="No datasets currently loaded.", icon='INFO')
            else:
                for ds_obj, arr in data_store.items():
                    row2 = datasets_box.row()
                    row2.label(text=f"{ds_obj.name}  shape={arr.shape}", icon='MESH_CUBE')
                    op = row2.operator("scene.unload_dataset", text="", icon='X')
                    op.box_name = ds_obj.name
            datasets_box.operator(
                "scene.unload_all_datasets", text="Unload All Datasets", icon='TRASH'
            )
        layout.separator()

        # --- Segmentation loading ---
        layout.prop(scene, "tissue_cartography_segmentation_file")
        row_seg = layout.row()
        row_seg.prop(scene, "tissue_cartography_segmentation_resolution")
        row_seg.prop(scene, "tissue_cartography_segmentation_axis_order")
        row_seg_smooth = layout.row()
        row_seg_smooth.prop(scene, "tissue_cartography_segmentation_sigma")
        layout.operator("scene.load_segmentation",
                        text="Get mesh(es) from binary segmentation .tiff file(s)")
        layout.label(
            text=(f"Loaded Segmentation Shape: {scene.tissue_cartography_segmentation_shape}. "
                  f"Loaded Segmentation Channels: {scene.tissue_cartography_segmentation_channels}")
        )
        layout.separator(type='LINE')
        layout.label(text="SELECTED DATASETS", icon='PINNED')
        row_sel = layout.row()
        row_sel.prop(scene, "tissue_cartography_active_box")
        row_sel.prop(scene, "tissue_cartography_active_mesh")
        layout.separator(type='LINE')
        layout.label(text="DATA VISUALIZATION", icon='HIDE_OFF')
        # --- Ortho-slice ---
        split_slice = layout.split(factor=0.2)
        axis_row = split_slice.row()
        axis_row.prop(scene, "tissue_cartography_slice_axis", expand=True)
        rest_row = split_slice.row()
        rest_row.prop(scene, "tissue_cartography_slice_position_pct", slider=True)
        rest_row.prop(scene, "tissue_cartography_slice_position")
        rest_row.prop(scene, "tissue_cartography_slice_channel")
        slice_row = layout.row()
        slice_row.prop(scene, "tissue_cartography_slice_show_intersection")
        slice_row.operator("scene.create_slice_plane", text="Create slice plane")
        layout.separator()

        # --- Vertex shader ---
        split_vertex = layout.split(factor=0.3)
        row_vertex = split_vertex.row()
        row_vertex.prop(scene, "tissue_cartography_vertex_shader_offset")
        row_vertex_ch = split_vertex.row()
        row_vertex_ch.prop(scene, "tissue_cartography_vertex_shader_channel_RGB")
        row_vertex2 = layout.row()
        row_vertex2.prop(scene, "tissue_cartography_vertex_shader_create_material")
        row_vertex2.operator("scene.vertex_shader", text="Initialize/refresh vertex shading")
        layout.separator(type='LINE')
        layout.label(text="PROJECTIONS", icon='IMAGE_DATA')
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
        row_batch.prop(scene, "tissue_cartography_batch_single_mesh")
        layout.label(
            text="Tip: Test resolution and axis order with a single mesh before batch processing.",
            icon='INFO',
        )
        layout.operator("scene.batch_projection", text="Batch Process And Save")
        layout.separator(type='LINE')
        layout.label(text="MESH REGISTRATION", icon='CON_SHRINKWRAP')
        # --- Alignment ---
        layout.prop(scene, "tissue_cartography_align_reference")
        row_align = layout.row()
        row_align.prop(scene, "tissue_cartography_prealign")
        row_align.prop(scene, "tissue_cartography_prealign_shear")
        row_align.prop(scene, "tissue_cartography_prealign_scale")
        row_align2 = layout.row()
        row_align2.prop(scene, "tissue_cartography_align_type")
        row_align2.prop(scene, "tissue_cartography_align_iter")
        layout.operator("scene.align", text="Align Meshes")
        layout.separator()

        # --- Shrinkwrap ---
        row_shrink = layout.row()
        row_shrink.prop(scene, "tissue_cartography_shrinkwrap_smooth")
        row_shrink.prop(scene, "tissue_cartography_shrinkwrap_iterative")
        layout.operator("scene.shrinkwrap", text="Shrinkwrap Reference to Selected")
        layout.separator()

        layout.operator("scene.help_popup", text="Show help", icon='HELP')
