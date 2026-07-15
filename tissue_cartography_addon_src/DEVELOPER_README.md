# Blender Tissue Cartography Add-on — Developer Reference

## Purpose

Tissue cartography extracts curved, 2d surfaces from volumetric microscopy and maps the image data to the plane
The purpose of this project is to leverage Blender's capabilities, notably the UV editor, for cartography and 
visualization.

The Blender add-on provides an interactive GUI for the tissue cartography pipeline: loading volumetric TIFF images and segmentations, generating and registering meshes, and baking image data onto UV-unwrapped surfaces. It shares algorithmic logic with the standalone Python library (`blender_tissue_cartography/`), but is self-contained. The add-on relies on Blender's UV unwrapping and is designed for interactive, single-dataset workflows. The standalone library targets automated batch pipelines.

The `bpy` library allows the add-on to interact with blender. It is only available within blender's python scripting interface, which is why you cannot run the add-on in a normal python interpreter. See [https://docs.blender.org/manual/en/latest/advanced/scripting/addon_tutorial.html](this tutorial).

---

## Module Structure

| File | Role |
|---|---|
| `__init__.py` | Addon entry point. Registers all operators and the UI panel. Defines all `bpy.props.SceneProperty` parameters. Hosts two-way slice-position sync callbacks. `register()` initializes the add-on.|
| `operators.py` | 12 `bpy.types.Operator` subclasses. `_run_projection()` is a shared helper encapsulating the full bake pipeline, called by both single and batch projection operators. |
| `ux.py` | Single `TissueCartographyPanel` with 6 collapsible sections (Data Loading, Loaded Datasets, Visualization, Projections, Mesh Registration, Misc). |
| `io_utils.py` | TIFF loading, axis-order string parsing, quantile normalization. |
| `mesh_utils.py` | Mesh creation from numpy arrays, numpy array serialization to/from Blender custom properties (`set_numpy_attribute` / `get_numpy_attribute`). |
| `projection.py` | UV baking: `get_uv_normal_world_per_loop()`, `bake_volumetric_data_to_uv()`, and `chunked_interpn()` for memory-safe interpolation. |
| `visualization.py` | Bounding boxes, ortho-slice planes, materials (slice, multi-layer projection, vertex color), and edge-length utilities for anti-aliasing. |
| `alignment.py` | Pure numpy: inertia-based pre-alignment + ICP (`combined_alignment()`). No Blender dependency — usable standalone. |

---

## Add-on design

- **Mesh-data association** To allow load multiple 3D datasets and meshes into the same blender file, image data is associated with mesh objects. Which data any operation is applied to is determined by the currently selected mesh. Loaded volumetric image data is associated with a `BoundingBox`, showing the volume covered by the image data.

- **Global 3D data dictionary.** `bpy.types.Scene.tissue_cartography_3D_data` is a plain Python dict (not a Blender property) keyed by bounding-box objects, and holds all loaded 3D images as `numpy` arrays in memory. Blender has no native format for large multidimensional arrays. This dict is not persisted in `.blend` files; users must reload TIFFs after reopening a file.

- **Numpy arrays serialized on objects.** Baked projections, resolution arrays, and other results are stored as binary custom properties via `set_numpy_attribute` / `get_numpy_attribute` (encoding: `(bytes, shape, dtype_str)` tuple). These *do* persist in `.blend` saves.

- **Axis convention.** All TIFF data is normalized to `(channels, x, y, z)` order immediately on load by `_normalize_tiff_axes()`. All downstream code assumes this layout.

- **Interactive slider for position selection.** To visualize the 3D data, users can select a 2D slice with
an interactive slider. The slider is implemented as follows. Slice position is exposed as both a 0–1 fraction and a µm value. Callbacks `_sync_pct_to_um` / `_sync_um_to_pct` keep them synchronized; a module-level `_slice_sync_lock` boolean prevents recursive callback loops.

- **Material creation.** To visualize image data in blender, the add-on creates materials
that show the image data on the mesh. For instance, `create_material_from_multilayer_array()` loads all channel/layer textures into the node graph but only connects channel 0 / layer 0 to the BSDF. Additional textures are available for manual rewiring in Blender's Shader Editor.

### Algorithm details

- **Multi-layer projection.** `bake_volumetric_data_to_uv()` accepts a list of normal offsets, sampling image intensity at multiple depths along surface normals in one pass. Output shape is `(channels, layers, res, res)`.

- **Chunked interpolation.** `chunked_interpn()` in `projection.py` avoids out-of-memory errors on large volumes by recursively splitting the longest axis with overlapping margins before calling `scipy.interpolate.interpn`.

- **Two-stage mesh alignment.** `combined_alignment()` runs an inertia-tensor pre-alignment (exhaustively tries all 8 axis-flip combinations to handle reflective ambiguity) followed by ICP. Either stage can be disabled independently.

- **Mesh–plane intersection visualization.** When "Show intersection" is checked and a slice plane is created, `create_intersection_line_visualization()` in `visualization.py` creates a Blender CURVE object (named `Intersect_<plane_name>`) that draws the intersection of the surface mesh with the cutting plane as a thick red line. This allows the user to check
if the surface mesh and the image are correctly aligned. A `depsgraph_update_post` handler (`_update_intersection_handler`) keeps it live: whenever the surface mesh or the slice plane is moved, the curve is recomputed.

  - **Analytic edge–plane intersection.** `compute_plane_intersection_segments()` computes the intersection of all mesh polygon s with the image plane. Works on non-watertight meshes (no Boolean intersection required).

  - **Global state.** Two key module-level variables in `visualization.py`:
    - `_intersection_trackers: dict` — maps each curve object's name to `(surface_mesh_name, slice_plane_name)`. Entries are removed automatically when either tracked object is deleted.
    - `_is_updating: bool` — re-entrancy guard. The `depsgraph_update_post` handler sets this to `True` while it updates curve splines; writing to curve data triggers another depsgraph update, so without this guard the handler would recurse infinitely.

  - **Handler lifecycle.** `_update_intersection_handler` is appended to `bpy.app.handlers.depsgraph_update_post` the first time an intersection curve is created, and is removed automatically once `_intersection_trackers` is empty or `unregister()` is called. 

---

## Building the Installable Extension

From the `tissue_cartography_addon_src/` directory:

```sh
"/Applications/Blender 5.2.app/Contents/MacOS/Blender" --command extension build \
  --source-dir . --output-dir ../blender_addon_download/ --split-platforms
```

This produces platform-specific `.zip` files in `../blender_addon_download/`.

### Re-downloading Wheels

The add-on targets **Blender 5.2 LTS**, which ships **Python 3.13**, so all
binary wheels must be cp313 (the wheel Python tag must always match the
Python version shipped by the targeted Blender). The `wheels/` directory must
contain platform wheels for `windows-x64`, `linux-x64`, and `macos-arm64`
(see `blender_manifest.toml` for the full list). To refresh:

```sh
PKGS="scipy==1.15.1 scikit-image==0.25.1 pillow==11.1.0 tifffile==2025.1.10 \
  imageio==2.37.0 networkx==3.4.2 lazy_loader==0.4 packaging==24.2"
pip download $PKGS --no-deps --dest wheels/ --python-version 313 --only-binary=:all: \
  --platform macosx_11_0_arm64 --platform macosx_12_0_arm64
pip download $PKGS --no-deps --dest wheels/ --python-version 313 --only-binary=:all: \
  --platform manylinux_2_17_x86_64 --platform manylinux2014_x86_64
pip download $PKGS --no-deps --dest wheels/ --python-version 313 --only-binary=:all: \
  --platform win_amd64
```

If you change package versions, update the filenames in
`blender_manifest.toml` accordingly. numpy is bundled with Blender and is
intentionally excluded (hence `--no-deps`).

---

## Interactive Testing

The `tissue_cartography_addon_src/` folder is symlinked into Blender's extension directory.
On a Mac, this means:

```sh
ln -s /path/to/blender_addon \
  ~/Library/Application\ Support/Blender/5.2/extensions/user_default/blender_tissue_cartography_interactive
```

After editing source files, reload the add-on without restarting Blender by running `reload_script.py` in Blender's **Scripting** workspace.