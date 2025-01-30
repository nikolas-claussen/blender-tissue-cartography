bl_info = {
    "name": "Tissue Cartography Packaged",
    "blender": (4, 3, 0),
    "category": "Scene",
}

import bpy
from bpy.props import StringProperty, FloatVectorProperty, IntVectorProperty, FloatProperty, IntProperty, BoolProperty, EnumProperty
from . import tissue_cartography
    

def register():
    """Add the add-on to the blender user interface"""
    bpy.utils.register_class(tissue_cartography.TissueCartographyPanel)
    bpy.utils.register_class(tissue_cartography.LoadTIFFOperator)
    bpy.utils.register_class(tissue_cartography.LoadSegmentationTIFFOperator)
    bpy.utils.register_class(tissue_cartography.CreateProjectionOperator)
    bpy.utils.register_class(tissue_cartography.SaveProjectionOperator)
    bpy.utils.register_class(tissue_cartography.BatchProjectionOperator)
    bpy.utils.register_class(tissue_cartography.SlicePlaneOperator)
    bpy.utils.register_class(tissue_cartography.VertexShaderInitializeOperator)
    bpy.utils.register_class(tissue_cartography.VertexShaderRefreshOperator)
    bpy.utils.register_class(tissue_cartography.AlignOperator)
    bpy.utils.register_class(tissue_cartography.ShrinkwrapOperator)
    bpy.utils.register_class(tissue_cartography.HelpPopupOperator)
    
    bpy.types.Scene.tissue_cartography_file = StringProperty(
        name="File Path",
        description="Path to the TIFF file",
        subtype='FILE_PATH',
    )
    bpy.types.Scene.tissue_cartography_resolution = FloatVectorProperty(
        name="x/y/z Resolution (µm)",
        description="Resolution in microns along x, y, z axes",
        size=3,
        default=(1.0, 1.0, 1.0),
    )
    bpy.types.Scene.tissue_cartography_axis_order= IntVectorProperty(
        name="Axis order",
        description="Permute axes after loading image. Use if loaded image shape appears incorrect. (0, 1, 2, 3) =  no permutation. Channel axis should be first axis!",
        size=4,
        default=(0, 1, 2, 3),
        min=0,
        max=3,
    )
    bpy.types.Scene.tissue_cartography_image_channels = IntProperty(
        name="Image Channels",
        description="Channels of the loaded image (read-only)",
        default=0,
        min=0,
    )
    bpy.types.Scene.tissue_cartography_image_shape = StringProperty(
        name="Image Shape",
        description="Shape of the loaded image (read-only)",
        default="Not loaded"
    )
    bpy.types.Scene.tissue_cartography_segmentation_file = StringProperty(
        name="Segmentation File Path",
        description="Path to the segmentation TIFF file. Should have values between 0-1. Selecting a folder instead of a single file will batch-process the full folder.",
        subtype='FILE_PATH',
    )
    bpy.types.Scene.tissue_cartography_segmentation_resolution = FloatVectorProperty(
        name="Segmentation x/y/z Resolution (µm)",
        description="Resolution of segmentation in microns along x, y, z axes",
        size=3,
        default=(1.0, 1.0, 1.0),
    )
    bpy.types.Scene.tissue_cartography_segmentation_sigma = FloatProperty(
        name="Segmentation Smoothing (µm)",
        description="Smothing kernel for extracting mesh from segmentation, in µm",
        default=0,
        min=0
    ) 
    bpy.types.Scene.tissue_cartography_slice_axis = EnumProperty(
        name="Slice Axis",
        description="Choose an axis",
        items=[('x', "X-Axis", "Align to the X axis"),
               ('y', "Y-Axis", "Align to the Y axis"),
               ('z', "Z-Axis", "Align to the Z axis")],
        default='x'
    )
    bpy.types.Scene.tissue_cartography_slice_position = FloatProperty(
        name="Slice Position (µm)",
        description="Position along the selected axis in µm",
        default=0
    )
    bpy.types.Scene.tissue_cartography_slice_channel = IntProperty(
        name="Slice Channel",
        description="Channel for slice plane",
        default=0,
        min=0,
    )
    bpy.types.Scene.tissue_cartography_vertex_shader_offset = FloatProperty(
        name="Vertex Shader Normal Offset (µm)",
        description="Normal offse for vertex shading.",
        default=0,
    )
    bpy.types.Scene.tissue_cartography_vertex_shader_channel = IntProperty(
        name="Vertex Shader Channel",
        description="Channel for vertex shading.",
        default=0,
        min=0,
    )
    bpy.types.Scene.tissue_cartography_offsets = StringProperty(
        name="Normal Offsets (µm)",
        description="Comma-separated list of floats for multilayer projection offsets",
        default="0",
    )
    bpy.types.Scene.tissue_cartography_projection_resolution = IntProperty(
        name="Projection Format (Pixels)",
        description="Resolution for the projection (e.g., 1024 for 1024x1024 pixels)",
        default=1024,
        min=1,
    )
    
    bpy.types.Scene.tissue_cartography_batch_directory = StringProperty(
        name="Batch Process Input Directory",
        description="Path to TIFF files directory",
        subtype='DIR_PATH',
    )
    bpy.types.Scene.tissue_cartography_batch_output_directory = StringProperty(
        name="Batch Process Output Directory",
        description="Path to TIFF files directory",
        subtype='DIR_PATH',
    )
    bpy.types.Scene.tissue_cartography_batch_create_materials = BoolProperty(
        name="Create materials",
        description="Enable or disable creating materials with projected texture in batch mode. Enabling can result in large .blend files.",
        default=True
    )
    bpy.types.Scene.tissue_cartography_prealign = BoolProperty(
        name="Pre-align?",
        description="Enable or disable pre-alignment. Do not use if the two meshes are already closely aligned.",
        default=True
    )
    bpy.types.Scene.tissue_cartography_prealign_shear = BoolProperty(
        name="Allow shear",
        description="Allow shear transformation during alignment.",
        default=True
    )
    bpy.types.Scene.tissue_cartography_align_type = EnumProperty(
        name="Align Mode",
        description="Choose an axis",
        items=[('selected', "Selected to Active", "Align selected meshes to active mesh."),
               ('active', "Active to Selected", "Align active mesh to selected meshe (creates copies of active mesh).")],
        default='selected'
    )
    bpy.types.Scene.tissue_cartography_align_iter = IntProperty(
        name="Iterations",
        description="ICP iterations during alignment.",
        default=100,
        min=1,
    )
    bpy.types.Scene.tissue_cartography_shrinkwarp_smooth = IntProperty(
        name="Shrinkwrap Corrective Smooth",
        description="Corrective smooth iterations during shrink-wrapping.",
        default=10,
        min=0,
    )
    bpy.types.Scene.tissue_cartography_shrinkwarp_iterative = EnumProperty(
        name="Shrinkwrap Mode",
        description="Choose an axis",
        items=[('one-to-all', "One-To-All", "Shrink-wrap active mesh to each selected individually"),
               ('forward', "Iterative Forward", "Shrink-wrap active mesh to selected meshes iteratively, starting with alpha-numerically first"),
               ('backward', "Iterative Backward", "Shrink-wrap active mesh to selected meshes iteratively, starting with alpha-numerically last")],
        default='one-to-all'
    )
    
    bpy.types.Scene.tissue_cartography_interpolators = dict()

def unregister():
    bpy.utils.unregister_class(tissue_cartography.TissueCartographyPanel)
    bpy.utils.unregister_class(tissue_cartography.LoadTIFFOperator)
    bpy.utils.unregister_class(tissue_cartography.LoadSegmentationTIFFOperator)
    bpy.utils.unregister_class(tissue_cartography.CreateProjectionOperator)
    bpy.utils.unregister_class(tissue_cartography.BatchProjectionOperator)
    bpy.utils.unregister_class(tissue_cartography.SaveProjectionOperator)
    bpy.utils.unregister_class(tissue_cartography.SlicePlaneOperator)
    bpy.utils.unregister_class(tissue_cartography.VertexShaderInitializeOperator)
    bpy.utils.unregister_class(tissue_cartography.VertexShaderRefreshOperator)
    bpy.utils.unregister_class(tissue_cartography.AlignOperator)
    bpy.utils.unregister_class(tissue_cartography.ShrinkwrapOperator)
    bpy.utils.unregister_class(tissue_cartography.HelpPopupOperator)

    del bpy.types.Scene.tissue_cartography_file 
    del bpy.types.Scene.tissue_cartography_resolution
    del bpy.types.Scene.tissue_cartography_axis_order
    del bpy.types.Scene.tissue_cartography_image_shape
    del bpy.types.Scene.tissue_cartography_segmentation_file
    del bpy.types.Scene.tissue_cartography_segmentation_resolution 
    del bpy.types.Scene.tissue_cartography_segmentation_sigma
    del bpy.types.Scene.tissue_cartography_offsets 
    del bpy.types.Scene.tissue_cartography_projection_resolution 
    del bpy.types.Scene.tissue_cartography_slice_axis 
    del bpy.types.Scene.tissue_cartography_slice_position 
    del bpy.types.Scene.tissue_cartography_slice_channel 
    del bpy.types.Scene.tissue_cartography_vertex_shader_offset 
    del bpy.types.Scene.tissue_cartography_vertex_shader_channel
    del bpy.types.Scene.tissue_cartography_prealign 
    del bpy.types.Scene.tissue_cartography_prealign_shear
    del bpy.types.Scene.tissue_cartography_align_iter
    del bpy.types.Scene.tissue_cartography_align_type
    del bpy.types.Scene.tissue_cartography_batch_directory
    del bpy.types.Scene.tissue_cartography_batch_output_directory
    del bpy.types.Scene.tissue_cartography_batch_create_materials
    del bpy.types.Scene.tissue_cartography_shrinkwarp_smooth
    del bpy.types.Scene.tissue_cartography_shrinkwarp_iterative
    
    if bpy.types.Scene.tissue_cartography_interpolators in globals():
        del bpy.types.Scene.tissue_cartography_interpolators
