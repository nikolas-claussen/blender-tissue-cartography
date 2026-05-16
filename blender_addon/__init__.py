bl_info = {
    "name": "Tissue Cartography (V5)",
    "blender": (4, 2, 0),
    "category": "Scene",
}

import bpy
from bpy.props import (
    StringProperty, FloatProperty, FloatVectorProperty,
    IntProperty, IntVectorProperty, BoolProperty, EnumProperty,
)

from .operators import OPERATOR_CLASSES
from .ux import TissueCartographyPanel


def register():
    """Register all add-on classes and scene properties."""
    for cls in OPERATOR_CLASSES:
        bpy.utils.register_class(cls)
    bpy.utils.register_class(TissueCartographyPanel)

    # In-memory store for 3D image data, keyed by bounding-box object.
    bpy.types.Scene.tissue_cartography_3D_data = {}

    # --- 3D image ---
    bpy.types.Scene.tissue_cartography_file = StringProperty(
        name="File Path",
        description="Path to the TIFF file",
        subtype='FILE_PATH',
    )
    bpy.types.Scene.tissue_cartography_resolution = FloatVectorProperty(
        name="x/y/z Resolution (µm)",
        description="Resolution in µm along x, y, z axes",
        size=3,
        default=(1.0, 1.0, 1.0),
    )
    bpy.types.Scene.tissue_cartography_axis_order = StringProperty(
        name="Axis order",
        description=(
            "Axis order: xyz/cxyz or any permutation. "
            "Leave empty to infer automatically."
        ),
        default="",
    )
    bpy.types.Scene.tissue_cartography_image_channels = IntProperty(
        name="Image Channels",
        description="#channels in loaded image (read-only)",
        default=0,
        min=0,
    )
    bpy.types.Scene.tissue_cartography_image_shape = StringProperty(
        name="Image Shape",
        description="Shape of loaded image (read-only)",
        default="Not loaded",
    )

    # --- Segmentation ---
    bpy.types.Scene.tissue_cartography_segmentation_file = StringProperty(
        name="Segmentation File Path",
        description=(
            "Path to segmentation TIFF file (values 0-1). "
            "Select a folder to batch-process."
        ),
        subtype='FILE_PATH',
    )
    bpy.types.Scene.tissue_cartography_segmentation_resolution = FloatVectorProperty(
        name="Segmentation x/y/z Resolution (µm)",
        description="Resolution of segmentation in µm along x, y, z axes",
        size=3,
        default=(1.0, 1.0, 1.0),
    )
    bpy.types.Scene.tissue_cartography_segmentation_axis_order = StringProperty(
        name="Axis order (segmentation)",
        description=(
            "Axis order for segmentation: xyz/cxyz or any permutation. "
            "Different channels represent different labels. Leave empty to infer."
        ),
        default="",
    )
    bpy.types.Scene.tissue_cartography_segmentation_sigma = FloatProperty(
        name="Smoothing (µm)",
        description="Gaussian smoothing kernel for mesh extraction, in µm",
        default=0.0,
        min=0.0,
    )
    bpy.types.Scene.tissue_cartography_segmentation_channels = IntProperty(
        name="Segmentation Channels",
        description="#channels in loaded segmentation (read-only)",
        default=0,
        min=0,
    )
    bpy.types.Scene.tissue_cartography_segmentation_shape = StringProperty(
        name="Segmentation Shape",
        description="Shape of loaded segmentation (read-only)",
        default="Not loaded",
    )

    # --- Ortho-slice ---
    bpy.types.Scene.tissue_cartography_slice_axis = EnumProperty(
        name="Slice Axis",
        description="Axis along which to slice",
        items=[
            ('x', "X-Axis", "Slice along the X axis"),
            ('y', "Y-Axis", "Slice along the Y axis"),
            ('z', "Z-Axis", "Slice along the Z axis"),
        ],
        default='x',
    )
    bpy.types.Scene.tissue_cartography_slice_position = FloatProperty(
        name="Slice Position (µm)",
        description="Position along the selected axis in µm",
        default=0.0,
    )
    bpy.types.Scene.tissue_cartography_slice_channel = IntProperty(
        name="Slice Channel",
        description="Image channel to display on the slice plane",
        default=0,
        min=0,
    )

    # --- Vertex shader ---
    bpy.types.Scene.tissue_cartography_vertex_shader_offset = FloatProperty(
        name="Vertex Shader Normal Offset (µm)",
        description="Normal offset for vertex shading",
        default=0.0,
    )
    bpy.types.Scene.tissue_cartography_vertex_shader_channel_RGB = IntVectorProperty(
        name="Vertex Shader Channels (RGB)",
        description="Image channels mapped to R, G, B of the vertex shader",
        default=(0, 0, 0),
        min=0,
        size=3,
    )
    bpy.types.Scene.tissue_cartography_vertex_shader_create_material = BoolProperty(
        name="Create new material?",
        description=(
            "Create a new vertex-color material. "
            "Uncheck to refresh the existing one."
        ),
        default=True,
    )

    # --- Projection ---
    bpy.types.Scene.tissue_cartography_offsets = StringProperty(
        name="Normal Offsets (µm)",
        description="Comma-separated list of floats for multi-layer projection offsets",
        default="0",
    )
    bpy.types.Scene.tissue_cartography_projection_resolution = IntProperty(
        name="Projection Format (Pixels)",
        description="Side length of the square projection texture in pixels",
        default=1024,
        min=1,
    )

    # --- Batch processing ---
    bpy.types.Scene.tissue_cartography_batch_directory = StringProperty(
        name="Batch Input Directory",
        description="Directory containing TIFF files for batch processing",
        subtype='DIR_PATH',
    )
    bpy.types.Scene.tissue_cartography_batch_output_directory = StringProperty(
        name="Batch Output Directory",
        description="Directory to write batch processing results",
        subtype='DIR_PATH',
    )
    bpy.types.Scene.tissue_cartography_batch_create_materials = BoolProperty(
        name="Create materials",
        description=(
            "Create projected-texture materials in batch mode. "
            "Enabling this can produce large .blend files."
        ),
        default=True,
    )

    # --- Alignment ---
    bpy.types.Scene.tissue_cartography_prealign = BoolProperty(
        name="Pre-align?",
        description=(
            "Pre-align by centroid and inertia axes before ICP. "
            "Disable if meshes are already closely aligned."
        ),
        default=True,
    )
    bpy.types.Scene.tissue_cartography_prealign_shear = BoolProperty(
        name="Allow shear",
        description="Allow shear transformation during pre-alignment",
        default=True,
    )
    bpy.types.Scene.tissue_cartography_prealign_scale = BoolProperty(
        name="Allow scale",
        description="Allow scale transformation during pre-alignment",
        default=True,
    )
    bpy.types.Scene.tissue_cartography_align_type = EnumProperty(
        name="Align Mode",
        description="Which mesh moves during alignment",
        items=[
            ('selected', "Selected to Active",
             "Align selected meshes to the active mesh"),
            ('active', "Active to Selected",
             "Align active mesh to each selected mesh (creates copies)"),
        ],
        default='selected',
    )
    bpy.types.Scene.tissue_cartography_align_iter = IntProperty(
        name="ICP Iterations",
        description="Number of ICP iterations during alignment (0 = skip ICP)",
        default=100,
        min=0,
    )

    # --- Shrinkwrap ---
    bpy.types.Scene.tissue_cartography_shrinkwrap_smooth = IntProperty(
        name="Shrinkwrap Corrective Smooth",
        description="Number of corrective smooth iterations during shrink-wrapping",
        default=2,
        min=0,
    )
    bpy.types.Scene.tissue_cartography_shrinkwrap_iterative = EnumProperty(
        name="Shrinkwrap Mode",
        description="How to iterate the shrinkwrap over multiple targets",
        items=[
            ('one-to-all', "One-To-All",
             "Shrink-wrap active mesh to each selected mesh independently"),
            ('forward', "Iterative Forward",
             "Shrink-wrap iteratively, alpha-numerically forward"),
            ('backward', "Iterative Backward",
             "Shrink-wrap iteratively, alpha-numerically backward"),
        ],
        default='one-to-all',
    )


def unregister():
    """Unregister all add-on classes and remove scene properties."""
    bpy.utils.unregister_class(TissueCartographyPanel)
    for cls in reversed(OPERATOR_CLASSES):
        bpy.utils.unregister_class(cls)

    props = [
        "tissue_cartography_file",
        "tissue_cartography_resolution",
        "tissue_cartography_axis_order",
        "tissue_cartography_image_channels",
        "tissue_cartography_image_shape",
        "tissue_cartography_segmentation_file",
        "tissue_cartography_segmentation_resolution",
        "tissue_cartography_segmentation_axis_order",
        "tissue_cartography_segmentation_sigma",
        "tissue_cartography_segmentation_channels",
        "tissue_cartography_segmentation_shape",
        "tissue_cartography_slice_axis",
        "tissue_cartography_slice_position",
        "tissue_cartography_slice_channel",
        "tissue_cartography_vertex_shader_offset",
        "tissue_cartography_vertex_shader_channel_RGB",
        "tissue_cartography_vertex_shader_create_material",
        "tissue_cartography_offsets",
        "tissue_cartography_projection_resolution",
        "tissue_cartography_batch_directory",
        "tissue_cartography_batch_output_directory",
        "tissue_cartography_batch_create_materials",
        "tissue_cartography_prealign",
        "tissue_cartography_prealign_shear",
        "tissue_cartography_prealign_scale",
        "tissue_cartography_align_type",
        "tissue_cartography_align_iter",
        "tissue_cartography_shrinkwrap_smooth",
        "tissue_cartography_shrinkwrap_iterative",
        "tissue_cartography_3D_data",
    ]
    for prop in props:
        if hasattr(bpy.types.Scene, prop):
            delattr(bpy.types.Scene, prop)

