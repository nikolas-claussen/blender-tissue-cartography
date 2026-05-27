"""
Blender Operators implementing all tissue cartography actions.
"""

import bpy
from bpy.types import Operator
import mathutils
import numpy as np
import tifffile
from pathlib import Path
from scipy import ndimage
from skimage import measure

from .io_utils import normalize_quantiles, axis_order_to_transpose
from .projection import (
    get_uv_mask,
    get_uv_normal_world_per_loop,
    bake_per_loop_values_to_uv,
    bake_volumetric_data_to_uv,
    chunked_interpn,
    compute_uv_area_distortion,
)
from .visualization import (
    create_box,
    create_slice_plane,
    create_intersection_line_visualization,
    get_slice_image,
    create_material_from_array,
    create_material_from_multilayer_array,
    compute_edge_lengths,
    assign_vertex_colors,
    create_vertex_color_material,
)
from .mesh_utils import (
    create_mesh_from_numpy,
    shrinkwrap_and_smooth,
    set_numpy_attribute,
    get_numpy_attribute,
)
from .alignment import combined_alignment


# ---------------------------------------------------------------------------
# Helpers shared by multiple operators
# ---------------------------------------------------------------------------

def _parse_axis_order(self, context, axis_order_string, data_shape):
    """Validate axis_order_string and return it, reporting errors via self."""
    if ''.join(sorted(axis_order_string)) not in ['', 'xyz', 'cxyz']:
        self.report({'ERROR'}, "Axis order must be empty, xyz, cxyz, or a permutation thereof.")
        return None
    if axis_order_string != '' and len(axis_order_string) != len(data_shape):
        self.report({'ERROR'}, "Number of axes in axis order does not match TIFF data.")
        return None
    return axis_order_string


def _normalize_tiff_axes(data, axis_order_string):
    """Apply axis order and ensure data has a leading channel axis."""
    if axis_order_string == '' and data.ndim == 4:
        channel_axis = int(np.argmin(data.shape))
        data = np.moveaxis(data, channel_axis, 0)
    if axis_order_string != '':
        data = data.transpose(axis_order_to_transpose(axis_order_string))
    if data.ndim == 3:
        data = data[np.newaxis]
    return data


def _run_projection(obj, data, box, projection_resolution, offsets_array):
    """
    Compute cartographic projection for a single (mesh, volumetric data) pair.

    Parameters
    ----------
    obj : bpy.types.Object
        Mesh object with an active UV map.
    data : np.ndarray, shape (C, X, Y, Z)
        Volumetric image data.
    box : bpy.types.Object
        Bounding-box object carrying the "resolution" attribute and world transform.
    projection_resolution : int
        UV image resolution in pixels.
    offsets_array : np.ndarray
        Normal offsets for the projection.

    Returns
    -------
    tuple of (baked_data, baked_normals, baked_world_positions, baked_local_resolution)
    """
    loop_uvs, loop_normals, loop_world_positions = get_uv_normal_world_per_loop(
        obj, filter_unique=True
    )
    baked_normals = bake_per_loop_values_to_uv(loop_uvs, loop_normals,
                                               image_resolution=projection_resolution)
    norms = np.linalg.norm(baked_normals, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    baked_normals = baked_normals / norms
    baked_world_positions = bake_per_loop_values_to_uv(loop_uvs, loop_world_positions,
                                                       image_resolution=projection_resolution)
    mask = get_uv_mask(obj, projection_resolution)
    baked_normals[~mask] = np.nan
    baked_world_positions[~mask] = np.nan
    baked_data = bake_volumetric_data_to_uv(
        data, baked_world_positions,
        get_numpy_attribute(box, "resolution"),
        baked_normals,
        normal_offsets=offsets_array,
        affine_matrix=np.linalg.inv(np.array(box.matrix_world)),
    )
    baked_local_resolution = compute_uv_area_distortion(obj, projection_resolution)
    baked_local_resolution[~mask] = np.nan
    return baked_data, baked_normals, baked_world_positions, baked_local_resolution


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

# --- I/O operators ---

class LoadTIFFOperator(Operator):
    """Load .tif file and resolution. Also creates a bounding box object."""
    bl_idname = "scene.load_tiff"
    bl_label = "Load TIFF File"

    def execute(self, context):
        file_path = bpy.path.abspath(context.scene.tissue_cartography_file)
        resolution = np.array(context.scene.tissue_cartography_resolution)
        self.report({'INFO'}, f"Resolution loaded: {resolution}")

        if not file_path.lower().endswith((".tif", ".tiff")):
            self.report({'ERROR'}, "Selected file is not a TIFF")
            return {'CANCELLED'}
        try:
            data = tifffile.imread(file_path)
            if data.ndim not in (3, 4):
                self.report({'ERROR'}, "Selected TIFF must have 3 or 4 axes.")
                return {'CANCELLED'}
            axis_order_string = context.scene.tissue_cartography_axis_order
            if _parse_axis_order(self, context, axis_order_string, data.shape) is None:
                return {'CANCELLED'}
            data = _normalize_tiff_axes(data, axis_order_string)

            context.scene.tissue_cartography_image_shape = str(data.shape[1:])
            context.scene.tissue_cartography_image_channels = data.shape[0]
            self.report({'INFO'}, f"TIFF file loaded with shape {data.shape}. ⚠️ Check that axis order is as expected! Else, adjust axis order field and reload.")
            context.scene.tissue_cartography_slice_extent = tuple(
                float(v) for v in np.array(data.shape[1:]) * resolution
            )

            box = create_box(*(np.array(data.shape[1:]) * resolution),
                             name=f"{Path(file_path).stem}_Box",
                             hide=False)
            box.display_type = 'WIRE'
            bpy.types.Scene.tissue_cartography_3D_data[box] = data
            set_numpy_attribute(box, "resolution", resolution)
            box["3D_data"] = True
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load TIFF file: {e}")
            return {'CANCELLED'}

        return {'FINISHED'}


class UnloadDatasetOperator(Operator):
    """Remove a single loaded 3D dataset from memory. The bounding box object is kept."""
    bl_idname = "scene.unload_dataset"
    bl_label = "Unload Dataset"

    box_name: bpy.props.StringProperty(name="Box Name", default="")

    def execute(self, context):
        data_store = bpy.types.Scene.tissue_cartography_3D_data
        box = bpy.data.objects.get(self.box_name)
        if box is None or box not in data_store:
            self.report({'WARNING'}, f"No loaded data found for '{self.box_name}'")
            return {'CANCELLED'}
        del data_store[box]
        self.report({'INFO'}, f"Unloaded data for '{self.box_name}'")
        return {'FINISHED'}


class UnloadAllDatasetsOperator(Operator):
    """Remove all loaded 3D datasets from memory."""
    bl_idname = "scene.unload_all_datasets"
    bl_label = "Unload All Datasets"

    def execute(self, context):
        count = len(bpy.types.Scene.tissue_cartography_3D_data)
        bpy.types.Scene.tissue_cartography_3D_data.clear()
        self.report({'INFO'}, f"Unloaded {count} dataset(s)")
        return {'FINISHED'}


class LoadSegmentationTIFFOperator(Operator):
    """
    Load segmentation .tif file and create a mesh from binary segmentation.

    Selecting a folder instead of a file batch-processes all TIFF files in the folder.
    """
    bl_idname = "scene.load_segmentation"
    bl_label = "Load Segmentation TIFF File"

    def execute(self, context):
        resolution_array = np.array(context.scene.tissue_cartography_segmentation_resolution)
        input_path = Path(bpy.path.abspath(context.scene.tissue_cartography_segmentation_file))
        if input_path.is_dir():
            files_to_process = [f for f in input_path.iterdir()
                                 if f.is_file() and f.suffix in (".tif", ".tiff")]
        elif input_path.is_file():
            files_to_process = [input_path]
        else:
            self.report({'ERROR'}, "Select a valid file or directory")
            return {'CANCELLED'}

        for i, file_path in enumerate(files_to_process):
            self.report({'INFO'}, f"Processing file {i + 1}/{len(files_to_process)}")
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
            try:
                data = tifffile.imread(file_path)
                if data.ndim not in (3, 4):
                    self.report({'ERROR'}, "Selected TIFF must have 3 or 4 axes.")
                    return {'CANCELLED'}
                axis_order_string = context.scene.tissue_cartography_segmentation_axis_order
                if _parse_axis_order(self, context, axis_order_string, data.shape) is None:
                    return {'CANCELLED'}
                data = _normalize_tiff_axes(data, axis_order_string)
                self.report({'INFO'}, f"TIFF file loaded with shape {data.shape}")
                context.scene.tissue_cartography_segmentation_shape = str(data.shape[1:])
                context.scene.tissue_cartography_segmentation_channels = data.shape[0]

                sigma = context.scene.tissue_cartography_segmentation_sigma
                for ic, channel in enumerate(data):
                    channel = channel.astype(float)
                    channel = (channel - channel.min()) / (channel.max() - channel.min())
                    if sigma > 0:
                        channel = ndimage.gaussian_filter(channel,
                                                          sigma=sigma / resolution_array)
                    verts, faces, _, _ = measure.marching_cubes(
                        channel, level=0.5, spacing=(1.0, 1.0, 1.0)
                    )
                    verts = verts * resolution_array
                    create_mesh_from_numpy(f"{file_path.stem}_c{ic}", verts, faces)
            except Exception as e:
                self.report({'ERROR'}, f"Failed to load segmentation: {e}")
                return {'CANCELLED'}
        return {'FINISHED'}


class SaveProjectionOperator(Operator):
    """
    Save cartographic projection to disk.
    
    Creates a folder with projected image data and surface geometry."""
    bl_idname = "scene.save_projection"
    bl_label = "Save Projection"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        obj = context.scene.tissue_cartography_active_mesh
        if obj is None:
            self.report({'ERROR'}, "Set an Active Mesh in the Selected Datasets section.")
            return {'CANCELLED'}
        if "baked_data" not in obj:
            self.report({'ERROR'}, "No baked data found on the active mesh. Run 'Create Projection' first!")
            return {'CANCELLED'}

        # Build output folder: {parent}/{name}/ with geometry subfolder
        name = Path(self.filepath).stem
        out_dir = Path(self.filepath).parent / name
        geom_dir = out_dir / f"{name}_geometry"
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            geom_dir.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(out_dir / f"{name}_ProjectedImageData.tif",
                             get_numpy_attribute(obj, "baked_data").transpose((1, 0, 2, 3)).astype(np.float32),
                             metadata={'axes': 'ZCYX'}, imagej=True)
            tifffile.imwrite(geom_dir / f"{name}_ProjectedNormals.tif",
                             get_numpy_attribute(obj, "baked_normals"))
            tifffile.imwrite(geom_dir / f"{name}_ProjectedPositions.tif",
                             get_numpy_attribute(obj, "baked_world_positions"))
            if "baked_local_resolution" in obj:
                tifffile.imwrite(geom_dir / f"{name}_LocalResolution.tif",
                                 get_numpy_attribute(obj, "baked_local_resolution"))
            else:
                self.report({'WARNING'}, "No local resolution data found. Re-run 'Create Projection' to compute it.")
            self.report({'INFO'}, f"Cartographic projection saved to {out_dir}")
        except Exception as e:
            self.report({'ERROR'}, f"Failed to save data: {e}")
            return {'CANCELLED'}
        return {'FINISHED'}


# --- Cartography operators ---

class CreateProjectionOperator(Operator):
    """
    Create a cartographic projection.

    Set the Active 3D Dataset and Active Mesh in the "Selected Datasets" section,
    then click this button to project the image data onto the mesh surface."""
    bl_idname = "scene.create_projection"
    bl_label = "Create Projection"

    def execute(self, context):
        box = context.scene.tissue_cartography_active_box
        obj = context.scene.tissue_cartography_active_mesh
        if box is None:
            self.report({'ERROR'}, "Set an Active 3D Dataset in the Selected Datasets section.")
            return {'CANCELLED'}
        if obj is None:
            self.report({'ERROR'}, "Set an Active Mesh in the Selected Datasets section.")
            return {'CANCELLED'}
        if not obj.data.uv_layers:
            self.report({'ERROR'}, "The selected mesh does not have a UV map!")
            return {'CANCELLED'}

        offsets_str = context.scene.tissue_cartography_offsets
        try:
            offsets_array = np.array([float(x) for x in offsets_str.split(",") if x.strip()])
            if offsets_array.size == 0:
                offsets_array = np.array([0.0])
            self.report({'INFO'}, f"Offsets loaded: {offsets_array}")
        except ValueError as e:
            self.report({'ERROR'}, f"Invalid offsets input: {e}")
            return {'CANCELLED'}
        set_numpy_attribute(obj, "projection_offsets", offsets_array)

        projection_resolution = context.scene.tissue_cartography_projection_resolution
        self.report({'INFO'}, f"Using projection resolution: {projection_resolution}")

        try:
            data = bpy.types.Scene.tissue_cartography_3D_data[box]
        except (KeyError, AttributeError):
            self.report({'ERROR'},
                        "Selected bounding box has no 3D data. Reload .tiff or select a different mesh.")
            return {'CANCELLED'}
        if not isinstance(data, np.ndarray) or data.ndim != 4:
            self.report({'ERROR'}, "Invalid 3D data array.")
            return {'CANCELLED'}

        baked_data, baked_normals, baked_world_positions, baked_local_resolution = _run_projection(
            obj, data, box, projection_resolution, offsets_array
        )
        set_numpy_attribute(obj, "baked_data", baked_data.astype(np.float32))
        set_numpy_attribute(obj, "baked_normals", baked_normals.astype(np.float32))
        set_numpy_attribute(obj, "baked_world_positions", baked_world_positions.astype(np.float32))
        set_numpy_attribute(obj, "baked_local_resolution", baked_local_resolution)
        create_material_from_multilayer_array(
            obj, baked_data, material_name=f"ProjectedMaterial_{obj.name}"
        )
        return {'FINISHED'}


class BatchProjectionOperator(Operator):
    """
    Batch-process cartographic projections.

    Meshes and TIFF files are matched by alphanumeric sort order: the first mesh (A→Z by name)
    is paired with the first file (A→Z by stem), and so on. The number of selected meshes must
    equal the number of TIFF files in the input directory.

    Single-mesh mode ("One mesh, many images"): uses the active mesh for every file in the
    input directory. Useful when the same surface is used across many image datasets.
    Material creation is skipped in this mode.

    The Active 3D Dataset (Selected Datasets section) provides the spatial reference
    (position/orientation) and resolution. Axis order is taken from the Data Loading section —
    test with a single mesh projection before running batch mode.
    """
    bl_idname = "scene.batch_projection"
    bl_label = "Create Projections (Batch Mode)"

    def execute(self, context):
        box = context.scene.tissue_cartography_active_box
        if box is None:
            self.report({'ERROR'}, "Set an Active 3D Dataset in the Selected Datasets section.")
            return {'CANCELLED'}

        batch_path = Path(bpy.path.abspath(context.scene.tissue_cartography_batch_directory))
        batch_out_path = Path(
            bpy.path.abspath(context.scene.tissue_cartography_batch_output_directory)
        )
        batch_files_sorted = sorted(
            [f for f in batch_path.iterdir()
             if f.suffix in (".tif", ".tiff") and "Baked" not in f.stem],
            key=lambda f: f.stem,
        )
        if not batch_files_sorted:
            self.report({'ERROR'}, "No TIFF files found in the Batch Input Directory.")
            return {'CANCELLED'}

        single_mesh_mode = context.scene.tissue_cartography_batch_single_mesh
        if single_mesh_mode:
            obj = context.active_object
            if obj is None or obj.type != 'MESH':
                self.report({'ERROR'}, "Set an active mesh object for single-mesh batch mode.")
                return {'CANCELLED'}
            pairs = [(obj, f) for f in batch_files_sorted]
        else:
            meshes_sorted = sorted(
                [o for o in context.selected_objects if o.type == 'MESH'],
                key=lambda o: o.name,
            )
            if len(meshes_sorted) != len(batch_files_sorted):
                self.report(
                    {'ERROR'},
                    f"Number of selected meshes ({len(meshes_sorted)}) does not match "
                    f"number of TIFF files ({len(batch_files_sorted)}). "
                    "Both are sorted alphanumerically and paired in order.",
                )
                return {'CANCELLED'}
            pairs = list(zip(meshes_sorted, batch_files_sorted))

        axis_order_string = context.scene.tissue_cartography_axis_order
        if ''.join(sorted(axis_order_string)) not in ('', 'xyz', 'cxyz'):
            self.report({'ERROR'}, "Axis order must be empty, xyz, cxyz, or a permutation thereof.")
            return {'CANCELLED'}

        offsets_str = context.scene.tissue_cartography_offsets
        try:
            offsets_array = np.array([float(x) for x in offsets_str.split(",") if x.strip()])
            if offsets_array.size == 0:
                offsets_array = np.array([0.0])
            self.report({'INFO'}, f"Offsets loaded: {offsets_array}")
        except ValueError as e:
            self.report({'ERROR'}, f"Invalid offsets input: {e}")
            return {'CANCELLED'}

        projection_resolution = context.scene.tissue_cartography_projection_resolution
        self.report({'INFO'}, f"Using projection resolution: {projection_resolution}")

        for iobj, (obj, file_path) in enumerate(pairs):
            self.report({'INFO'}, f"Processing {iobj + 1}/{len(pairs)}")
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
            if not obj.data.uv_layers:
                self.report({'ERROR'}, f"Mesh {obj.name} does not have a UV map!")
                return {'CANCELLED'}
            set_numpy_attribute(obj, "projection_offsets", offsets_array)
            try:
                data = tifffile.imread(file_path)
                if data.ndim not in (3, 4):
                    self.report({'ERROR'},
                                f"TIFF for {obj.name} must have 3 or 4 axes.")
                    return {'CANCELLED'}
                if _parse_axis_order(self, context, axis_order_string, data.shape) is None:
                    return {'CANCELLED'}
                data = _normalize_tiff_axes(data, axis_order_string)
            except Exception as e:
                self.report({'ERROR'}, f"Failed loading TIFF for {obj.name}: {e}")
                return {'CANCELLED'}

            baked_data, baked_normals, baked_world_positions, baked_local_resolution = _run_projection(
                obj, data, box, projection_resolution, offsets_array
            )
            out_stem = file_path.stem if single_mesh_mode else obj.name
            out_dir = batch_out_path / out_stem
            geom_dir = out_dir / f"{out_stem}_geometry"
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
                geom_dir.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(
                    out_dir / f"{out_stem}_BakedData.tif",
                    baked_data.astype(np.float32).transpose((1, 0, 2, 3)),
                    metadata={'axes': 'ZCYX'}, imagej=True,
                )
                tifffile.imwrite(
                    geom_dir / f"{out_stem}_BakedNormals.tif",
                    baked_normals.astype(np.float32),
                )
                tifffile.imwrite(
                    geom_dir / f"{out_stem}_BakedPositions.tif",
                    baked_world_positions.astype(np.float32),
                )
                tifffile.imwrite(
                    geom_dir / f"{out_stem}_LocalResolution.tif",
                    baked_local_resolution,
                )
                self.report({'INFO'}, f"Projection saved for {out_stem}")
            except Exception as e:
                self.report({'ERROR'}, f"Failed to save data for {out_stem}: {e}")
                return {'CANCELLED'}

            if not single_mesh_mode and context.scene.tissue_cartography_batch_create_materials:
                set_numpy_attribute(obj, "baked_data", baked_data.astype(np.float32))
                set_numpy_attribute(obj, "baked_normals", baked_normals.astype(np.float32))
                set_numpy_attribute(obj, "baked_world_positions",
                                    baked_world_positions.astype(np.float32))
                set_numpy_attribute(obj, "baked_local_resolution", baked_local_resolution)
                create_material_from_multilayer_array(
                    obj, baked_data, material_name=f"ProjectedMaterial_{obj.name}"
                )
        return {'FINISHED'}


# --- Visualization operators ---


class SlicePlaneOperator(Operator):
    """Create a slice plane along the selected axis with a texture from the Active 3D Dataset."""
    bl_idname = "scene.create_slice_plane"
    bl_label = "Create Slice Plane"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        box = context.scene.tissue_cartography_active_box
        if box is None:
            self.report({'ERROR'}, "Set an Active 3D Dataset in the Selected Datasets section.")
            return {'CANCELLED'}
        try:
            data = bpy.types.Scene.tissue_cartography_3D_data[box]
        except KeyError:
            self.report({'ERROR'},
                        "Selected bounding box has no 3D data. Reload .tiff or select a different box.")
            return {'CANCELLED'}
        resolution = get_numpy_attribute(box, "resolution")
        if not isinstance(data, np.ndarray) or data.ndim != 4:
            self.report({'ERROR'}, "Invalid 3D data array.")
            return {'CANCELLED'}
        ch = context.scene.tissue_cartography_slice_channel
        if ch >= data.shape[0]:
            self.report({'ERROR'}, f"Channel {ch} is out of bounds.")
            return {'CANCELLED'}

        show_intersection = context.scene.tissue_cartography_slice_show_intersection
        surface_mesh = context.scene.tissue_cartography_active_mesh
        if show_intersection and surface_mesh is None:
            self.report({'ERROR'}, "Set an Active Mesh in the Selected Datasets section to visualize intersection.")
            return {'CANCELLED'}

        length, width, height = np.array(data.shape[1:]) * resolution
        axis = context.scene.tissue_cartography_slice_axis
        position = context.scene.tissue_cartography_slice_position

        slice_plane = create_slice_plane(length, width, height, axis=axis, position=position)
        data_name = box.name[:-4]
        slice_plane.name = f"{slice_plane.name}_{data_name}"
        slice_plane.matrix_world = box.matrix_world

        slice_img = get_slice_image(data, resolution, axis=axis, position=position)
        slice_img = normalize_quantiles(slice_img, quantiles=(0.01, 0.99),
                                        channel_axis=0, clip=True)
        create_material_from_array(
            slice_plane, slice_img[ch],
            material_name=f"Slice_{data_name}_{axis}_{position}"
        )
        if show_intersection:
            create_intersection_line_visualization(slice_plane, surface_mesh)
        return {'FINISHED'}


class VertexShaderOperator(Operator):
    """Color mesh vertices according to 3D image intensity from the Active 3D Dataset.
    
    Input data is anti-aliased and interpolated at vertex positions. The resulting
    intensities are normalized to [0, 1] based on 1st and 99th percentiles.
    """
    bl_idname = "scene.vertex_shader"
    bl_label = "Initialize Vertex Shader"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        box = context.scene.tissue_cartography_active_box
        obj = context.scene.tissue_cartography_active_mesh
        if box is None:
            self.report({'ERROR'}, "Set an Active 3D Dataset in the Selected Datasets section.")
            return {'CANCELLED'}
        if obj is None:
            self.report({'ERROR'}, "Set an Active Mesh in the Selected Datasets section.")
            return {'CANCELLED'}
        try:
            data = bpy.types.Scene.tissue_cartography_3D_data[box]
        except KeyError:
            self.report({'ERROR'},
                        "Selected bounding box has no 3D data. Reload .tiff or select a different box.")
            return {'CANCELLED'}
        resolution = get_numpy_attribute(box, "resolution")
        if not isinstance(data, np.ndarray) or data.ndim != 4:
            self.report({'ERROR'}, "Invalid 3D data array.")
            return {'CANCELLED'}
        channels_rgb = list(context.scene.tissue_cartography_vertex_shader_channel_RGB)
        if any(c >= data.shape[0] for c in channels_rgb):
            self.report({'ERROR'}, "Channel(s) out of bounds for the data array.")
            return {'CANCELLED'}

        offset = context.scene.tissue_cartography_vertex_shader_offset
        positions = np.array([
            box.matrix_world.inverted() @ obj.matrix_world @ (v.co + offset * v.normal)
            for v in obj.data.vertices
        ])

        # anti-aliasing: use a scalar smoothing width (median edge length / resolution mean)
        median_edge = np.median(compute_edge_lengths(obj))
        aa_scale = float(1. * median_edge / np.mean(resolution))
        def anti_aliasing_filter(x):
            return ndimage.uniform_filter(x, size=max(1, int(round(aa_scale))))

        x, y, z = [np.arange(ni) for ni in data.shape[1:]]
        intensities = np.zeros((positions.shape[0], 3))
        for i, ic in enumerate(channels_rgb):
            intensities[:, i] = chunked_interpn(
                (x, y, z), data[ic], positions / resolution,
                chunk_size=100, overlap=10,
                method="linear", bounds_error=False,
                local_filter=anti_aliasing_filter,
            )

        #qmins = np.array([np.quantile(data[ic, ::4, ::4, ::4], 0.05) for ic in channels_rgb])
        #qmaxs = np.array([np.quantile(data[ic, ::4, ::4, ::4], 0.99) for ic in channels_rgb])
        intensities = np.nan_to_num(intensities)
        qmins, qmaxs = np.percentile(intensities, (1, 99), axis=0)
        denom = qmaxs - qmins
        denom[denom == 0] = 1.0
        intensities = np.clip((intensities - qmins) / denom, 0, 1)

        assign_vertex_colors(obj, intensities)
        if context.scene.tissue_cartography_vertex_shader_create_material:
            create_vertex_color_material(obj, material_name=f"VertexColorMaterial_{obj.name}")
        return {'FINISHED'}


# --- Mesh alignment operators ---

class AlignOperator(Operator):
    """Align meshes to a reference mesh by rotation, translation, and scaling."""
    bl_idname = "scene.align"
    bl_label = "Align Meshes"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        reference_mesh = context.scene.tissue_cartography_align_reference
        if reference_mesh is None or reference_mesh.type != 'MESH':
            self.report({'ERROR'}, "Set a valid Reference Mesh before aligning.")
            return {'CANCELLED'}

        others = [x for x in context.selected_objects
                  if x != reference_mesh and x.type == 'MESH']
        if not others:
            self.report({'ERROR'}, "Select at least one mesh (other than the reference) to align.")
            return {'CANCELLED'}

        ref_verts = np.array([reference_mesh.matrix_world @ v.co
                              for v in reference_mesh.data.vertices])

        if context.scene.tissue_cartography_align_type == "selected":
            # Align each selected mesh in-place to the reference.
            for mesh in others:
                self.report({'INFO'}, f"Aligning: {mesh.name} to {reference_mesh.name}")
                bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
                source = np.array([mesh.matrix_world @ v.co for v in mesh.data.vertices])
                trafo_matrix = combined_alignment(
                    source, ref_verts,
                    pre_align=context.scene.tissue_cartography_prealign,
                    shear=context.scene.tissue_cartography_prealign_shear,
                    scale=context.scene.tissue_cartography_prealign_scale,
                    iterations=context.scene.tissue_cartography_align_iter,
                )
                mesh.matrix_world = mathutils.Matrix(trafo_matrix) @ mesh.matrix_world

        elif context.scene.tissue_cartography_align_type == "active":
            # For each selected mesh, create an aligned copy of the reference.
            for target_mesh in others:
                target_name = target_mesh.name  # capture before copy may trigger Blender renaming
                self.report({'INFO'}, f"Aligning copy of {reference_mesh.name} to {target_name}")
                bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
                target = np.array([target_mesh.matrix_world @ v.co
                                   for v in target_mesh.data.vertices])
                trafo_matrix = combined_alignment(
                    ref_verts, target,
                    pre_align=context.scene.tissue_cartography_prealign,
                    shear=context.scene.tissue_cartography_prealign_shear,
                    scale=context.scene.tissue_cartography_prealign_scale,
                    iterations=context.scene.tissue_cartography_align_iter,
                )
                ref_copied = reference_mesh.copy()
                ref_copied.data = reference_mesh.data.copy()
                ref_copied.name = f"{target_name}_align"  # set before linking to prevent naming cascade
                bpy.context.collection.objects.link(ref_copied)
                ref_copied.matrix_world = (
                    mathutils.Matrix(trafo_matrix) @ reference_mesh.matrix_world
                )
        return {'FINISHED'}


class ShrinkwrapOperator(Operator):
    """Copy and shrink-wrap the reference mesh to each selected mesh. Set ICP iterations to 0 to skip registration."""
    bl_idname = "scene.shrinkwrap"
    bl_label = "Shrink-Wrap Reference to Selected"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        source_mesh = context.scene.tissue_cartography_align_reference
        if source_mesh is None or source_mesh.type != 'MESH':
            self.report({'ERROR'}, "Set a valid Reference Mesh before shrink-wrapping.")
            return {'CANCELLED'}

        mode = context.scene.tissue_cartography_shrinkwrap_iterative
        targets = sorted(
            [x for x in context.selected_objects if x != source_mesh and x.type == 'MESH'],
            key=lambda x: x.name,
        )
        if not targets:
            self.report({'ERROR'}, "Select at least one mesh (other than the reference) as a target.")
            return {'CANCELLED'}
        if mode == "backward":
            targets = targets[::-1]

        for target_mesh in targets:
            target_name = target_mesh.name  # capture before copy may trigger Blender renaming
            self.report({'INFO'}, f"Shrink-wrapping: {source_mesh.name} to {target_name}")
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

            if context.scene.tissue_cartography_align_iter > 0:
                target = np.array([target_mesh.matrix_world @ v.co
                                   for v in target_mesh.data.vertices])
                source = np.array([source_mesh.matrix_world @ v.co
                                   for v in source_mesh.data.vertices])
                trafo_matrix = combined_alignment(
                    source, target,
                    pre_align=context.scene.tissue_cartography_prealign,
                    shear=context.scene.tissue_cartography_prealign_shear,
                    scale=context.scene.tissue_cartography_prealign_scale,
                    iterations=context.scene.tissue_cartography_align_iter,
                )
            else:
                trafo_matrix = np.eye(4)

            source_mesh_copied = source_mesh.copy()
            source_mesh_copied.data = source_mesh.data.copy()
            source_mesh_copied.name = f"{target_name}_wrap"  # set before linking to prevent naming cascade
            bpy.context.collection.objects.link(source_mesh_copied)
            source_mesh_copied.matrix_world = (
                mathutils.Matrix(trafo_matrix) @ source_mesh.matrix_world
            )

            shrinkwrap_and_smooth(
                source_mesh_copied, target_mesh,
                corrective_smooth_iter=context.scene.tissue_cartography_shrinkwrap_smooth,
            )

            data_transfer = target_mesh.modifiers.new(name="DataTransfer", type='DATA_TRANSFER')
            data_transfer.object = source_mesh_copied
            data_transfer.use_loop_data = True
            data_transfer.data_types_loops = {'UV'}
            data_transfer.loop_mapping = 'POLYINTERP_NEAREST'

            original_active_obj = bpy.context.view_layer.objects.active
            bpy.context.view_layer.objects.active = target_mesh
            bpy.ops.object.datalayout_transfer(modifier="DataTransfer")
            bpy.ops.object.modifier_apply(modifier="DataTransfer")
            bpy.context.view_layer.objects.active = original_active_obj

            if mode in ("forward", "backward"):
                source_mesh = source_mesh_copied
        return {'FINISHED'}


class HelpPopupOperator(Operator):
    """Open the online help page."""
    bl_idname = "scene.help_popup"
    bl_label = "Tissue Cartography Help"

    def execute(self, context):
        bpy.ops.wm.url_open(
            url="https://nikolas-claussen.github.io/blender-tissue-cartography/"
        )
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Registration list (imported by __init__.py)
# ---------------------------------------------------------------------------

OPERATOR_CLASSES = [
    LoadTIFFOperator,
    LoadSegmentationTIFFOperator,
    CreateProjectionOperator,
    SaveProjectionOperator,
    BatchProjectionOperator,
    SlicePlaneOperator,
    VertexShaderOperator,
    AlignOperator,
    ShrinkwrapOperator,
    UnloadDatasetOperator,
    UnloadAllDatasetsOperator,
    HelpPopupOperator,
]
