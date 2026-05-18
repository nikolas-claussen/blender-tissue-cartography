"""
UV baking and volumetric-to-UV interpolation.

Converts 3D world-space data (positions, normals, image intensities) onto a 2D UV grid.
"""

import bpy
import numpy as np
from scipy import interpolate
from skimage import draw as skdraw


def get_uv_normal_world_per_loop(mesh_obj, filter_unique=False):
    """
    Get UV coordinates, normals, and world positions for each loop (half-edge).

    Parameters
    ----------
    mesh_obj : bpy.types.Object
        Mesh object with an active UV map.
    filter_unique : bool, optional
        If True, remove duplicate loops (identical UV, normal, and position).

    Returns
    -------
    loop_uvs : np.array of shape (n_loops, 2)
    loop_normals : np.array of shape (n_loops, 3)
    loop_world_positions : np.array of shape (n_loops, 3)
    """
    if not mesh_obj:
        raise TypeError("No object selected")
    if mesh_obj.type != 'MESH':
        raise TypeError("Selected object is not a mesh")
    world_matrix = mesh_obj.matrix_world
    uv_layer = mesh_obj.data.uv_layers.active
    if not uv_layer:
        raise RuntimeError("Mesh does not have an active UV map")

    n = len(mesh_obj.data.loops)
    n_verts = len(mesh_obj.data.vertices)

    loop_vert_flat = np.zeros(n, dtype=np.int32)
    mesh_obj.data.loops.foreach_get("vertex_index", loop_vert_flat)

    uv_flat = np.zeros(n * 2, dtype=np.float32)
    uv_layer.data.foreach_get("uv", uv_flat)
    loop_uvs = uv_flat.reshape(n, 2)

    vert_normals_flat = np.zeros(n_verts * 3, dtype=np.float32)
    mesh_obj.data.vertices.foreach_get("normal", vert_normals_flat)
    mat3 = np.array(world_matrix.to_3x3())
    loop_normals = vert_normals_flat.reshape(n_verts, 3)[loop_vert_flat] @ mat3.T

    vert_co_flat = np.zeros(n_verts * 3, dtype=np.float32)
    mesh_obj.data.vertices.foreach_get("co", vert_co_flat)
    vert_co = vert_co_flat.reshape(n_verts, 3)
    translation = np.array(world_matrix.translation)
    loop_world_positions = vert_co[loop_vert_flat] @ mat3.T + translation

    if filter_unique:
        unique_loops = np.unique(np.hstack([loop_uvs, loop_normals, loop_world_positions]), axis=0)
        loop_uvs = unique_loops[:, :2]
        loop_normals = unique_loops[:, 2:5]
        loop_world_positions = unique_loops[:, 5:]

    norms = np.linalg.norm(loop_normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    loop_normals = np.round(loop_normals / norms, decimals=4)
    return loop_uvs, loop_normals, loop_world_positions


def get_uv_area_per_loop(mesh_obj):
    """
    Get per-triangle-corner UV coordinates and 3D/UV area ratios for a mesh.

    Analogous to get_uv_normal_world_per_loop: returns raw per-loop data suitable
    for passing to bake_per_loop_values_to_uv. Degenerate UV triangles (zero UV
    area) are assigned NaN and should be filtered before baking.

    Parameters
    ----------
    mesh_obj : bpy.types.Object
        Mesh object with an active UV map. Vertex coordinates should be in µm.

    Returns
    -------
    tri_loop_uvs : np.ndarray, shape (n_tris * 3, 2)
        UV coordinates for each triangle corner.
    loop_area_ratio : np.ndarray, shape (n_tris * 3,)
        3D area (µm²) per UV area (UV-unit²) for the triangle containing each corner.
        NaN for degenerate UV triangles.
    """
    uv_layer = mesh_obj.data.uv_layers.active
    if not uv_layer:
        raise RuntimeError("Mesh does not have an active UV map")

    mesh_obj.data.calc_loop_triangles()
    n_tris = len(mesh_obj.data.loop_triangles)
    if n_tris == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    n_loops = len(mesh_obj.data.loops)
    n_verts = len(mesh_obj.data.vertices)

    uv_flat = np.zeros(n_loops * 2, dtype=np.float32)
    uv_layer.data.foreach_get("uv", uv_flat)
    loop_uvs_all = uv_flat.reshape(n_loops, 2)

    tri_loops_flat = np.zeros(n_tris * 3, dtype=np.int32)
    mesh_obj.data.loop_triangles.foreach_get("loops", tri_loops_flat)
    tri_verts_flat = np.zeros(n_tris * 3, dtype=np.int32)
    mesh_obj.data.loop_triangles.foreach_get("vertices", tri_verts_flat)

    vert_co_flat = np.zeros(n_verts * 3, dtype=np.float32)
    mesh_obj.data.vertices.foreach_get("co", vert_co_flat)
    mat3 = np.array(mesh_obj.matrix_world.to_3x3())
    translation = np.array(mesh_obj.matrix_world.translation)
    vert_world = vert_co_flat.reshape(n_verts, 3) @ mat3.T + translation

    tri_uvs = loop_uvs_all[tri_loops_flat].reshape(n_tris, 3, 2)
    tri_pos = vert_world[tri_verts_flat].reshape(n_tris, 3, 3)

    e1_3d = tri_pos[:, 1] - tri_pos[:, 0]
    e2_3d = tri_pos[:, 2] - tri_pos[:, 0]
    area_3d = 0.5 * np.linalg.norm(np.cross(e1_3d, e2_3d), axis=-1)

    e1_uv = tri_uvs[:, 1] - tri_uvs[:, 0]
    e2_uv = tri_uvs[:, 2] - tri_uvs[:, 0]
    area_uv = 0.5 * np.abs(e1_uv[:, 0] * e2_uv[:, 1] - e1_uv[:, 1] * e2_uv[:, 0])

    with np.errstate(invalid='ignore', divide='ignore'):
        area_ratio = np.where(area_uv > 0, area_3d / area_uv, np.nan)

    tri_loop_uvs = loop_uvs_all[tri_loops_flat]        # (n_tris * 3, 2)
    loop_area_ratio = np.repeat(area_ratio, 3)         # (n_tris * 3,)
    return tri_loop_uvs, loop_area_ratio


def chunked_interpn(points, values, xi, method='linear', bounds_error=True,
                    fill_value=np.nan, chunk_size=100, overlap=5, local_filter=None):
    """
    Multidimensional regular grid interpolation for large datasets by splitting into chunks.

    Chunking can drastically improve speed and memory footprint for large,
    high-dimensional (>2d) data. Chunking is likely inefficient for small arrays and
    small numbers of sample positions.

    Uses scipy.interpolate.interpn on each chunk; can be used as a drop-in replacement.
    Works on *rectilinear* grids (rectangular grid with even or uneven spacing).

    Parameters
    ----------
    points : tuple of ndarray of float, with shapes (m1,), ..., (mn,)
        The points defining the regular grid in n dimensions.
    values : array_like, shape (m1, ..., mn, ...)
        Data on the regular grid. Complex data is accepted.
    xi : ndarray of shape (..., ndim)
        Coordinates to sample at.
    method : str, optional
        Interpolation method: "linear", "nearest", "slinear", "cubic", "quintic",
        "pchip", or "splinef2d" (2D only).
    bounds_error : bool, optional
        If True, raise ValueError for out-of-domain requests.
    fill_value : number, optional
        Value for out-of-domain points when bounds_error is False.
    chunk_size : int, optional
        Chunk size (hyper-cubes of side chunk_size + 2*overlap).
    overlap : int, optional
        Overlap between chunks. Defaults to 5.
    local_filter : callable or None
        Filter applied to each chunk before interpolation (e.g. smoothing).

    Returns
    -------
    ndarray, shape xi.shape[:-1] + values.shape[ndim:]
        Interpolated values at xi.
    """
    xi_shape = xi.shape
    xi = xi.reshape((-1, xi_shape[-1]))
    results = fill_value * np.zeros(xi.shape[0])
    chunk_axis = np.argmax(values.shape)
    if values.shape[chunk_axis] <= (chunk_size + 2 * overlap):
        if local_filter is not None:
            values = local_filter(values)
        results = interpolate.interpn(points, values, xi,
                                      method=method, bounds_error=bounds_error,
                                      fill_value=fill_value)
        return results.reshape(xi_shape[:-1])
    splits = np.array([
        points[chunk_axis][min((i + 1) * chunk_size, points[chunk_axis].shape[0] - 1)]
        for i in range(int(np.ceil(values.shape[chunk_axis] / chunk_size)))
    ])
    chunk_per_position = np.searchsorted(splits, xi[:, chunk_axis], side="right")
    for i in range(int(np.ceil(values.shape[chunk_axis] / chunk_size))):
        mask = (chunk_per_position == i)
        if not mask.any():
            continue
        start = max(i * chunk_size - overlap, 0)
        stop = (i + 1) * chunk_size + overlap
        slices = tuple(
            slice(start, stop) if j == chunk_axis else slice(None)
            for j in range(values.ndim)
        )
        chunk = values[slices]
        chunk_points = [
            (p if j != chunk_axis else p[start:stop]) for j, p in enumerate(points)
        ]
        results[mask] = chunked_interpn(
            chunk_points, chunk, xi[mask],
            chunk_size=chunk_size, overlap=overlap, local_filter=local_filter,
            method=method, bounds_error=bounds_error, fill_value=fill_value,
        )
    return results.reshape(xi_shape[:-1])


def bake_per_loop_values_to_uv(loop_uvs, loop_values, image_resolution):
    """
    Bake (interpolate) per-loop values onto a uniform UV grid.

    UV coordinates outside [0, 1] are ignored.

    Parameters
    ----------
    loop_uvs : np.array of shape (n_loops, 2)
        UV coordinates of each loop.
    loop_values : np.array of shape (n_loops, ...)
        Values to bake. Can have any trailing shape (scalar or vector field).
    image_resolution : int
        Size of the UV grid (number of pixels per side).

    Returns
    -------
    np.array of shape (image_resolution, image_resolution, ...)
        Field across the [0, 1]^2 UV grid. Positions without data are np.nan.
    """
    U, V = np.meshgrid(*(2 * (np.linspace(0, 1, image_resolution),)))
    interpolated = interpolate.griddata(loop_uvs, loop_values, (U, V), method='linear')[::-1]
    return interpolated


def bake_volumetric_data_to_uv(image, baked_world_positions, resolution, baked_normals,
                                normal_offsets=(0,), affine_matrix=None):
    """
    Interpolate volumetric image data onto a UV coordinate grid.

    Uses baked 3D world positions corresponding to each UV grid point.
    3D coordinates (in microns) are converted to image coordinates via the resolution
    scaling factor. Providing multiple normal_offsets produces a multi-layer pullback.

    Parameters
    ----------
    image : np.array of shape (n_channels, nx, ny, nz)
        Volumetric image; axis 0 is the channel axis.
    baked_world_positions : np.array of shape (image_resolution, image_resolution, 3)
        3D world positions baked to UV grid. Positions without data are np.nan.
    resolution : np.array of shape (3,)
        Voxel size in microns for each spatial axis.
    baked_normals : np.array of shape (image_resolution, image_resolution, 3)
        3D normals baked to UV grid. Positions without data are np.nan.
    normal_offsets : array-like of shape (n_layers,), default (0,)
        Offsets along the surface normal in microns. 0 = no shift.
    affine_matrix : np.array of shape (4, 4) or None
        If not None, transform coordinates by this affine before interpolation.

    Returns
    -------
    np.array of shape (n_channels, n_layers, image_resolution, image_resolution)
        Multi-layer volumetric data baked onto the UV grid.
    """
    x, y, z = [np.arange(ni) for ni in image.shape[1:]]
    res = baked_world_positions.shape[0]
    baked_data = np.zeros(shape=(image.shape[0], len(normal_offsets), res, res))
    positions = np.stack(
        [(baked_world_positions + o * baked_normals) for o in normal_offsets], axis=0
    )
    if affine_matrix is not None:
        positions = np.einsum('ij,nxyj->nxyi', affine_matrix[:3, :3], positions)
        positions = positions + affine_matrix[:3, 3]
    positions = positions / resolution
    for ic, channel in enumerate(image):
        baked_data[ic] = chunked_interpn(
            (x, y, z), channel, positions,
            chunk_size=100, overlap=2, method="linear",
            bounds_error=False, local_filter=None,
        )
    return baked_data


def get_uv_mask(mesh_obj, image_resolution):
    """
    Get UV coverage mask for a mesh object as a boolean np.array.

    Rasterizes the mesh UV triangles using skimage.draw.polygon.
    No files are written to disk; no edit-mode switch or selection side-effects.

    Parameters
    ----------
    mesh_obj : bpy.types.Object
        Mesh object with an active UV map.
    image_resolution : int
        Width/height of the output image in pixels.

    Returns
    -------
    np.array of shape (image_resolution, image_resolution), dtype bool
        True where the UV layout has coverage.
    """

    uv_layer = mesh_obj.data.uv_layers.active
    if not uv_layer:
        raise RuntimeError("Mesh does not have an active UV map")

    # VERIFY (Blender 4.1+): calc_loop_triangles() may be a no-op; safe to call regardless.
    mesh_obj.data.calc_loop_triangles()
    n_tris = len(mesh_obj.data.loop_triangles)
    if n_tris == 0:
        return np.zeros((image_resolution, image_resolution), dtype=bool)

    # Read all UV coordinates at once via foreach_get (fast, avoids Python loop over loops).
    n_loops = len(mesh_obj.data.loops)
    uv_flat = np.zeros(n_loops * 2, dtype=np.float32)
    uv_layer.data.foreach_get("uv", uv_flat)
    loop_uvs = uv_flat.reshape(n_loops, 2)

    # Read all triangle loop indices at once into a flat int array.
    # VERIFY: MeshLoopTriangle.foreach_get("loops", flat_int32) should fill
    # [tri0_loop0, tri0_loop1, tri0_loop2, tri1_loop0, ...].
    tri_loops_flat = np.zeros(n_tris * 3, dtype=np.int32)
    mesh_obj.data.loop_triangles.foreach_get("loops", tri_loops_flat)
    # UV coords for every triangle corner: shape (n_tris, 3, 2)
    tri_uvs = loop_uvs[tri_loops_flat.reshape(n_tris, 3)]

    # Convert UV [0,1] → pixel indices.
    # u → column index, v → row index (row 0 = V=0, i.e. UV bottom, before [::-1] flip).
    res = image_resolution
    px = (tri_uvs * res).astype(np.int32).clip(0, res - 1)  # (n_tris, 3, 2)

    mask = np.zeros((res, res), dtype=bool)
    for i in range(n_tris):
        rr, cc = skdraw.polygon(px[i, :, 1], px[i, :, 0], shape=(res, res))
        mask[rr, cc] = True

    # Flip rows so row 0 = V=1, matching the convention of bake_per_loop_values_to_uv.
    return mask[::-1]


def compute_uv_area_distortion(mesh_obj, image_resolution):
    """
    Compute local resolution map (sqrt of 3D surface area per pixel) for a UV-unwrapped mesh.

    For each UV pixel, the local resolution is the square root of the 3D surface area (µm²)
    mapped to that pixel. Pixels outside the UV coverage are undefined.

    Parameters
    ----------
    mesh_obj : bpy.types.Object
        Mesh object with an active UV map. Vertex coordinates should be in µm.
    image_resolution : int
        UV image resolution in pixels.

    Returns
    -------
    np.ndarray of shape (image_resolution, image_resolution), dtype float32
        Local resolution in µm/pixel. Undefined outside UV coverage.
    """
    tri_loop_uvs, loop_area_ratio = get_uv_area_per_loop(mesh_obj)
    if tri_loop_uvs.shape[0] == 0:
        return np.full((image_resolution, image_resolution), np.nan, dtype=np.float32)
    valid = np.isfinite(loop_area_ratio)
    baked_ratio = bake_per_loop_values_to_uv(
        tri_loop_uvs[valid], loop_area_ratio[valid], image_resolution
    )
    local_resolution = (np.sqrt(np.abs(baked_ratio)) / image_resolution).astype(np.float32)
    return local_resolution