# Blender Tissue Cartography Add-on — Developer Reference

## Purpose

The Blender add-on provides an interactive GUI for the tissue cartography pipeline: loading volumetric TIFF images and segmentations, generating and registering meshes, and baking image data onto UV-unwrapped surfaces. It shares algorithmic logic with the standalone Python library (`blender_tissue_cartography/`), but is self-contained. The add-on relies on Blender's UV unwrapping and is designed for interactive, single-dataset workflows. The standalone library targets automated batch pipelines.

---

## Module Structure

| File | Role |
|---|---|
| `__init__.py` | Addon entry point. Registers all operators and the UI panel. Defines all `bpy.props.SceneProperty` parameters. Hosts two-way slice-position sync callbacks. |
| `operators.py` | 12 `bpy.types.Operator` subclasses. `_run_projection()` is a shared helper encapsulating the full bake pipeline, called by both single and batch projection operators. |
| `ux.py` | Single `TissueCartographyPanel` with 6 collapsible sections (Data Loading, Loaded Datasets, Visualization, Projections, Mesh Registration, Misc). |
| `io_utils.py` | TIFF loading, axis-order string parsing, quantile normalization. |
| `mesh_utils.py` | Mesh creation from numpy arrays, numpy array serialization to/from Blender custom properties (`set_numpy_attribute` / `get_numpy_attribute`). |
| `projection.py` | UV baking: `get_uv_normal_world_per_loop()`, `bake_volumetric_data_to_uv()`, and `chunked_interpn()` for memory-safe interpolation. |
| `visualization.py` | Bounding boxes, ortho-slice planes, materials (slice, multi-layer projection, vertex color), and edge-length utilities for anti-aliasing. |
| `alignment.py` | Pure numpy: inertia-based pre-alignment + ICP (`combined_alignment()`). No Blender dependency — usable standalone. |

---

## Key Design Decisions

- **Global 3D data dictionary.** `bpy.types.Scene.tissue_cartography_3D_data` is a plain Python dict (not a Blender property) keyed by bounding-box object, holding all loaded numpy arrays in memory. Blender has no native format for large multidimensional arrays. This dict is not persisted in `.blend` files; users must reload TIFFs after reopening a file.

- **Numpy arrays serialized on objects.** Baked projections, resolution arrays, and other results are stored as binary custom properties via `set_numpy_attribute` / `get_numpy_attribute` (encoding: `(bytes, shape, dtype_str)` tuple). These *do* persist in `.blend` saves.

- **Axis convention.** All TIFF data is normalized to `(channels, x, y, z)` order immediately on load by `_normalize_tiff_axes()`. All downstream code assumes this layout.

- **Chunked interpolation.** `chunked_interpn()` in `projection.py` avoids out-of-memory errors on large volumes by recursively splitting the longest axis with overlapping margins before calling `scipy.interpolate.interpn`.

- **Multi-layer projection.** `bake_volumetric_data_to_uv()` accepts a list of normal offsets, sampling image intensity at multiple depths along surface normals in one pass. Output shape is `(channels, layers, res, res)`.

- **Interactive slider for position selection.** To visualize the 3D data, users can select a 2D slice with
an interactive slider. The slider is implemented as follows. Slice position is exposed as both a 0–1 fraction and a µm value. Callbacks `_sync_pct_to_um` / `_sync_um_to_pct` keep them synchronized; a module-level `_slice_sync_lock` boolean prevents recursive callback loops.

- **Two-stage mesh alignment.** `combined_alignment()` runs an inertia-tensor pre-alignment (exhaustively tries all 8 axis-flip combinations to handle reflective ambiguity) followed by ICP. Either stage can be disabled independently.

- **Material creation.** To visualize image data in blender, the add-on creates materials
that show the image data on the mesh. For instance, `create_material_from_multilayer_array()` loads all channel/layer textures into the node graph but only connects channel 0 / layer 0 to the BSDF. Additional textures are available for manual rewiring in Blender's Shader Editor.

---

## Building the Installable Extension

From the `blender_addon/` directory:

```sh
/Applications/Blender.app/Contents/MacOS/Blender --command extension build \
  --source-dir . --output-dir addon_zips/ --split-platforms
```

This produces platform-specific `.zip` files in `addon_zips/`.

### Re-downloading Wheels

Blender 4.2+ uses **Python 3.11**. The `wheels/` directory must contain platform wheels for `windows-x64`, `linux-x64`, and `macos-arm64` (see `blender_manifest.toml` for the full list). To refresh:

```sh
pip download scipy scikit-image pillow tifffile imageio networkx lazy_loader packaging \
  --dest wheels/ --python-version 3.11 --only-binary=:all: \
  --platform <TAG>
```

Replace `<TAG>` with the target platform tag (e.g. `win_amd64`, `manylinux_2_17_x86_64`, `macosx_12_0_arm64`). Run once per platform. numpy is bundled with Blender and is intentionally excluded.

---

## Interactive Testing

The `blender_addon/` folder is symlinked into Blender's extension directory:

```sh
ln -s /path/to/blender_addon \
  ~/Library/Application\ Support/Blender/4.5/extensions/user_default/tissue_cartography
```

After editing source files, reload the add-on without restarting Blender by running `reload_script.py` in Blender's **Scripting** workspace. The script reloads all submodules in reverse-depth order, then disables and re-enables the add-on.