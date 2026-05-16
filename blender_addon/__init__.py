bl_info = {
    "name": "Tissue Cartography (V2)",
    "blender": (4, 2, 0),
    "category": "Scene",
}

import bpy
from bpy.props import (
    StringProperty, FloatProperty, FloatVectorProperty,
    IntProperty, IntVectorProperty, BoolProperty, EnumProperty, PointerProperty,
)

import numpy as np

from .operators import OPERATOR_CLASSES
from .ux import TissueCartographyPanel


# ---------------------------------------------------------------------------
# Slice-position sync helpers
# ---------------------------------------------------------------------------

_slice_sync_lock = False

_AXIS_INDEX = {'x': 0, 'y': 1, 'z': 2}


def _get_extent(context):
    """Return the per-axis physical extents (µm) for the active bounding box.

    Falls back to the stored scene property when no valid box is active.
    Never writes any scene property, so it is safe to call from draw().
    """
    from .mesh_utils import get_numpy_attribute
    active = context.active_object
    if active and "3D_data" in active:
        arr = bpy.types.Scene.tissue_cartography_3D_data.get(active)
        if arr is not None and arr.ndim == 4:
            res = get_numpy_attribute(active, "resolution")
            if res is not None:
                return np.array(arr.shape[1:]) * res
    return np.array(context.scene.tissue_cartography_slice_extent)


def _sync_pct_to_um(self, context):
    """Called when the fraction slider changes — update the µm field."""
    global _slice_sync_lock
    if _slice_sync_lock:
        return
    _slice_sync_lock = True
    try:
        extent = _get_extent(context)
        axis = context.scene.tissue_cartography_slice_axis
        max_um = extent[_AXIS_INDEX[axis]]
        if max_um > 0:
            context.scene.tissue_cartography_slice_position = (
                context.scene.tissue_cartography_slice_position_pct * max_um
            )
    finally:
        _slice_sync_lock = False


def _sync_um_to_pct(self, context):
    """Called when the µm field changes — update the fraction slider."""
    global _slice_sync_lock
    if _slice_sync_lock:
        return
    _slice_sync_lock = True
    try:
        extent = _get_extent(context)
        axis = context.scene.tissue_cartography_slice_axis
        max_um = extent[_AXIS_INDEX[axis]]
        if max_um > 0:
            context.scene.tissue_cartography_slice_position_pct = max(
                0.0, min(1.0, context.scene.tissue_cartography_slice_position / max_um)
            )
    finally:
        _slice_sync_lock = False


def _sync_axis_change(self, context):
    """Called when the slice axis changes — recompute µm from preserved fraction."""
    global _slice_sync_lock
    if _slice_sync_lock:
        return
    _slice_sync_lock = True
    try:
        extent = _get_extent(context)
        axis = context.scene.tissue_cartography_slice_axis
        max_um = extent[_AXIS_INDEX[axis]]
        if max_um > 0:
            context.scene.tissue_cartography_slice_position = (
                context.scene.tissue_cartography_slice_position_pct * max_um
            )
    finally:
        _slice_sync_lock = False


def register():
    """Register all add-on classes and scene properties."""
    for cls in OPERATOR_CLASSES:
        bpy.utils.register_class(cls)
    bpy.utils.register_class(TissueCartographyPanel)

    # In-memory store for 3D image data, keyed by bounding-box object.
    bpy.types.Scene.tissue_cartography_3D_data = {}

    bpy.types.Scene.tissue_cartography_show_datasets = BoolProperty(
        name="Show Loaded Datasets",
        description="Expand/collapse the loaded datasets list",
        default=False,
    )

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
    bpy.types.Scene.tissue_cartography_slice_extent = FloatVectorProperty(
        name="Slice Extent (µm)",
        description="Physical size of the loaded image per axis (set automatically on TIFF load)",
        size=3,
        default=(100.0, 100.0, 100.0),
        options={'HIDDEN'},
    )
    bpy.types.Scene.tissue_cartography_slice_axis = EnumProperty(
        name="Axis",
        description="Axis along which to slice",
        items=[
            ('x', "X", "Slice along the X axis"),
            ('y', "Y", "Slice along the Y axis"),
            ('z', "Z", "Slice along the Z axis"),
        ],
        default='x',
        update=_sync_axis_change,
    )
    bpy.types.Scene.tissue_cartography_slice_position_pct = FloatProperty(
        name="Slice Position",
        description="Position along the selected axis as a fraction of the full extent (drag slider)",
        default=0.0,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
        update=_sync_pct_to_um,
    )
    bpy.types.Scene.tissue_cartography_slice_position = FloatProperty(
        name="µm",
        description="Position along the selected axis in µm (type for precise entry)",
        default=0.0,
        min=0.0,
        update=_sync_um_to_pct,
    )
    bpy.types.Scene.tissue_cartography_slice_channel = IntProperty(
        name="Channel",
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
    bpy.types.Scene.tissue_cartography_batch_single_mesh = BoolProperty(
        name="One mesh, many images",
        description=(
            "Use the active mesh for every file in the input directory. "
            "Useful when the same surface is used across many image datasets. "
            "Material creation is skipped in this mode."
        ),
        default=False,
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
    bpy.types.Scene.tissue_cartography_active_box = PointerProperty(
        name="Active 3D Dataset",
        description="BoundingBox object whose 3D image data is used for visualization and projection",
        type=bpy.types.Object,
        poll=lambda self, obj: "3D_data" in obj,
    )
    bpy.types.Scene.tissue_cartography_active_mesh = PointerProperty(
        name="Active Mesh",
        description="Mesh surface to project onto or shade",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'MESH' and "3D_data" not in obj,
    )
    bpy.types.Scene.tissue_cartography_align_reference = PointerProperty(
        name="Reference Mesh",
        description="Reference mesh to align to (or to copy from)",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'MESH',
    )
    bpy.types.Scene.tissue_cartography_align_type = EnumProperty(
        name="Align Mode",
        description="Which mesh moves during alignment",
        items=[
            ('selected', "Align Selected to Reference",
             "Align all selected meshes in-place to the reference mesh"),
            ('active', "Align Reference Copy to Each Selected",
             "For each selected mesh, create an aligned copy of the reference mesh"),
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
    for cls in [TissueCartographyPanel] + list(reversed(OPERATOR_CLASSES)):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass

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
        "tissue_cartography_slice_extent",
        "tissue_cartography_slice_axis",
        "tissue_cartography_slice_position_pct",
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
        "tissue_cartography_batch_single_mesh",
        "tissue_cartography_active_box",
        "tissue_cartography_active_mesh",
        "tissue_cartography_prealign",
        "tissue_cartography_prealign_shear",
        "tissue_cartography_prealign_scale",
        "tissue_cartography_align_reference",
        "tissue_cartography_align_type",
        "tissue_cartography_align_iter",
        "tissue_cartography_shrinkwrap_smooth",
        "tissue_cartography_shrinkwrap_iterative",
        "tissue_cartography_3D_data",
        "tissue_cartography_show_datasets",
    ]
    for prop in props:
        if hasattr(bpy.types.Scene, prop):
            delattr(bpy.types.Scene, prop)

