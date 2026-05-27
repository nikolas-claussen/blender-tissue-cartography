# PyMeshLab Remeshing — Developer Notes

## Overview

Standalone Blender 4.2+ extension exposing five PyMeshLab mesh processing operations:

| Operation | Filter | Use case |
|---|---|---|
| Isotropic Remeshing | `meshing_isotropic_explicit_remeshing` | Uniform triangle quality after marching-cubes |
| Quadric Decimation | `meshing_decimation_quadric_edge_collapse` | Reduce face count |
| Screened Poisson | `generate_surface_reconstruction_screened_poisson` | Surface from point cloud |
| Alpha Wrap | `generate_alpha_wrap` | Watertight hull of non-manifold mesh |
| Mesh Cleanup | multiple | Remove degenerate elements |

All operations work **in-memory** (see below).

## Files

```
pymeshlab_remesh_addon/
  blender_manifest.toml   Blender 4.2+ extension metadata, wheels list
  __init__.py             Registration: scene properties + classes
  mesh_utils.py           In-memory Blender ↔ pymeshlab conversion
  operators.py            Five Blender operators
  ux.py                   Properties-panel UI
  wheels/                 pymeshlab wheels (not in git, regenerate with download_wheels.sh)
  download_wheels.sh      Fetches pymeshlab wheels from PyPI
```

## Building

From the `pymeshlab_remesh_addon/` directory:

```sh
# 1. Ensure wheels are present
./download_wheels.sh

# 2. Build with Blender's extension tool
/Applications/Blender.app/Contents/MacOS/Blender \
    --command extension build \
    --source-dir . \
    --output-dir ../blender_addon_download/ \
    --split-platforms
```

## In-memory mesh conversion

`mesh_utils.py` mirrors `blender_tissue_cartography/interface_pymeshlab.py`:

- **Blender → pymeshlab**: `mesh.calc_loop_triangles()` → numpy arrays via
  `foreach_get` → `pymeshlab.Mesh(vertex_matrix=..., face_matrix=...)`.
  For point clouds (no faces), omits `face_matrix` and includes
  `v_normals_matrix` so Poisson reconstruction can reuse existing normals.

- **pymeshlab → Blender**: `ms.current_mesh().vertex_matrix()/.face_matrix()`
  → `mesh.from_pydata()` → new linked object.

## Poisson reconstruction from a Blender point cloud

Blender stores point-cloud-like data in mesh objects with vertices but no
polygons (i.e. `len(mesh.loop_triangles) == 0`). The pipeline is:

1. `blender_to_pymeshlab` detects no faces → passes only `vertex_matrix` +
   `v_normals_matrix` (existing vertex normals) to `pymeshlab.Mesh`.
2. `compute_normal_for_point_clouds(k=k, smoothiter=2)` recomputes/smooths
   normals from k-nearest-neighbour geometry (toggle with *Compute Normals* checkbox).
3. `generate_surface_reconstruction_screened_poisson(depth=d, fulldepth=fd)`
   reconstructs the watertight surface.
4. `pymeshlab_to_blender` reads back the new mesh layer as a Blender object.

The *Compute Normals* option is on by default. Disable it only if the input
already has well-oriented normals and you want to skip the neighbourhood step.

## PyMeshLab version and wheel notes

The bundled wheels are **pymeshlab 2025.7.post1** (cp311, released January 2026).

**Why not an older release?**
`2023.12.post2` (and `post3`) contain a confirmed Windows crash bug: calling
`load_default_plugins()` on import triggers an access violation on Python 3.11+
(see [PyMeshLab issue #398](https://github.com/cnr-isti-vclab/PyMeshLab/issues/398)).
`2025.7.post1` is retained for macOS/Linux compatibility with Blender 4.5's
Python 3.11 runtime, but it does not solve the Windows native crash.

**Windows — not supported** — pymeshlab is unconditionally blocked on Windows via the
shared runtime guard in `pymeshlab_runtime.py` (`sys.platform == 'win32'`). The panel
shows an error message and all operators return `{'CANCELLED'}` without ever calling
`import pymeshlab`.

On supported platforms (macOS/Linux), `pymeshlab` is imported once at module-load
time in `operators.py`, `mesh_utils.py`, and `ux.py` and the import status is cached.

The reason is a native crash that Python `try/except` *cannot* intercept: pymeshlab's
plugin manager (`PyInit_pmeshlab → PluginManager::loadPlugins`) loads
`filter_texture_defragmentation.dll`, which in turn loads `Qt5Core.dll` late into the
Blender process. Qt5's thread-local-storage (TLS) initialisation then dereferences a
null pointer inside `MSVCP140.dll::Thrd_yield`, killing the process with
`EXCEPTION_ACCESS_VIOLATION`. This affects every tested pymeshlab version including
`2025.7.post1`. See [pymeshlab issue #398](https://github.com/cnr-isti-vclab/PyMeshLab/issues/398).

**Linux glibc requirement** — the `manylinux_2_35` wheels require **glibc 2.35+**
(Ubuntu 22.04 / Fedora 36 or newer). Users on older distributions (Ubuntu 20.04,
glibc 2.31) will see the panel warning rather than a crash, but the add-on
will not function. If broad older-Linux support is needed in the future, the
macOS wheel can stay on `2025.7.post1` while the Linux wheel is
downgraded to a `manylinux_2_31` build (the last one available is
`2023.12.post3`, which is unaffected by the Windows crash).
