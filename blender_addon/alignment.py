"""
Point-cloud alignment: ICP, Procrustes, centroid/inertia-based pre-alignment.

No bpy dependency.
"""

import numpy as np
import itertools
from scipy import linalg, spatial, stats


def package_affine_transformation(matrix, vector):
    """Package matrix transformation & translation into (d+1, d+1) affine matrix."""
    matrix_rep = np.hstack([matrix, vector[:, np.newaxis]])
    matrix_rep = np.pad(matrix_rep, ((0, 1), (0, 0)), constant_values=0)
    matrix_rep[-1, -1] = 1
    return matrix_rep


def get_inertia(pts):
    """Get inertia tensor of 3d point cloud."""
    pts_nomean = pts - np.mean(pts, axis=0)
    x, y, z = pts_nomean.T
    Ixx = np.mean(x**2)
    Ixy = np.mean(x * y)
    Ixz = np.mean(x * z)
    Iyy = np.mean(y**2)
    Iyz = np.mean(y * z)
    Izz = np.mean(z * z)
    return np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])


def align_by_centroid_and_inertia(source, target, scale=True, shear=True, improper=False,
                                   n_samples=10000):
    """
    Align source point cloud to target point cloud using affine transformation.

    Align by matching centroids and axes of inertia tensor. Since the inertia tensor is invariant
    under reflections along its principal axes, all 2^3 reflections are tried and the one leading
    to the best agreement with the target is chosen.

    Parameters
    ----------
    source : np.array of shape (n_source, 3)
        Point cloud to be aligned.
    target : np.array of shape (n_target, 3)
        Point cloud to align to.
    scale : bool, default True
        Whether to allow scale transformation (True) or rotations only (False).
    shear : bool, default False
        Whether to allow shear transformation (True) or rotations/scale only (False).
    improper : bool, default False
        Whether to allow transformations with determinant -1.
    n_samples : int, optional
        Number of samples of source to use when estimating distances.

    Returns
    -------
    affine_matrix_rep : np.array of shape (4, 4)
        Affine transformation source -> target.
    aligned : np.array of shape (n_source, 3)
        Aligned coordinates.
    """
    target_centroid = np.mean(target, axis=0)
    target_inertia = get_inertia(target)
    target_eig = np.linalg.eigh(target_inertia)

    source_centroid = np.mean(source, axis=0)
    source_inertia = get_inertia(source)
    source_eig = np.linalg.eigh(source_inertia)

    flips = [np.diag([i, j, k]) for i, j, k in itertools.product(*(3 * [[-1, 1]]))]
    trafo_matrix_candidates = []
    tree = spatial.cKDTree(target)
    samples = source[np.random.randint(low=0, high=source.shape[0],
                                       size=min([n_samples, source.shape[0]])), :]
    distances = []
    for flip in flips:
        if shear:
            trafo_matrix = (source_eig.eigenvectors
                            @ np.diag(np.sqrt(target_eig.eigenvalues / source_eig.eigenvalues))
                            @ flip @ target_eig.eigenvectors.T)
        elif scale and not shear:
            scale_fact = np.sqrt(
                stats.gmean(target_eig.eigenvalues) / stats.gmean(source_eig.eigenvalues))
            trafo_matrix = scale_fact * source_eig.eigenvectors @ flip @ target_eig.eigenvectors.T
        else:
            trafo_matrix = source_eig.eigenvectors @ flip @ target_eig.eigenvectors.T
        if not improper and np.linalg.det(trafo_matrix) < 0:
            continue
        trafo_matrix = trafo_matrix.T
        trafo_matrix_candidates.append(trafo_matrix)
        trafo_translate = target_centroid - trafo_matrix @ source_centroid
        aligned = samples @ trafo_matrix.T + trafo_translate
        distances.append(np.mean(tree.query(aligned)[0]))
    trafo_matrix = trafo_matrix_candidates[np.argmin(distances)]
    print('inferred rotation/scale', trafo_matrix)
    trafo_translate = target_centroid - trafo_matrix @ source_centroid
    aligned = source @ trafo_matrix.T + trafo_translate
    affine_matrix_rep = package_affine_transformation(trafo_matrix, trafo_translate)
    print('inferred translation', trafo_translate)
    return affine_matrix_rep, aligned


def procrustes(source, target, scale=True):
    """
    Procrustes analysis, a similarity test for two data sets.

    Each input matrix is a set of points or vectors (the rows of the matrix).
    Procrustes standardizes both such that tr(AA^T) = 1 and both sets are centered
    around the origin, then applies the optimal transform to the source matrix to
    minimize pointwise squared differences.

    Parameters
    ----------
    source : array_like
        Matrix, n rows represent points in k (columns) space.
        The data from source will be transformed to fit the pattern in target.
    target : array_like
        Matrix, n rows represent points in k (columns) space.
        target is the reference data.
    scale : bool, default True
        Whether to allow scaling transformations.

    Returns
    -------
    trafo_affine : np.array of shape (4, 4)
        Affine transformation from source to target.
    aligned : array_like
        The orientation of source that best fits target.
    disparity : float
        np.linalg.norm(aligned - target, axis=1).mean()
    """
    mtx1 = np.array(target, dtype=np.float64, copy=True)
    mtx2 = np.array(source, dtype=np.float64, copy=True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    centroid1, centroid2 = (np.mean(mtx1, 0), np.mean(mtx2, 0))
    mtx1 -= centroid1
    mtx2 -= centroid2

    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)
    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")
    mtx1 /= norm1
    mtx2 /= norm2

    R, s = linalg.orthogonal_procrustes(mtx1, mtx2)
    mtx2 = np.dot(mtx2, R.T) * s

    aligned = norm1 * mtx2 + centroid1
    disparity = np.mean(np.linalg.norm(aligned - target, axis=1))

    if scale:
        trafo_matrix = (norm1 / norm2) * s * R
    else:
        trafo_matrix = (norm1 / norm2) * R
    trafo_translate = centroid1 - trafo_matrix @ centroid2
    trafo_affine = package_affine_transformation(trafo_matrix, trafo_translate)
    return trafo_affine, aligned, disparity


def icp(source, target, initial=None, threshold=1e-4, max_iterations=20, scale=True,
        n_samples=1000):
    """
    Apply the iterative closest point algorithm to align point cloud source to target.

    Will only produce reasonable results if the initial transformation is roughly correct.
    Initial transformation can be found by applying Procrustes' analysis to landmark points
    or by inertia+centroid based alignment (align_by_centroid_and_inertia).

    Parameters
    ----------
    source : (n, 3) float
        Source points in space.
    target : (m, 3) float
        Target points in space.
    initial : (4, 4) float or None
        Initial transformation.
    threshold : float
        Stop when change in cost is less than threshold.
    max_iterations : int
        Maximum number of iterations.
    scale : bool, optional
        Whether to allow dilations. If False, orthogonal procrustes is used.
    n_samples : int or None
        If not None, n_samples sample points are randomly chosen from source array.

    Returns
    -------
    matrix : (4, 4) float
        The transformation matrix sending source to target.
    transformed : (n, 3) float
        The image of source under the transformation.
    cost : float
        The cost of the transformation.
    """
    total_matrix = np.eye(4) if initial is None else initial
    tree = spatial.cKDTree(target)
    samples = (source[np.random.randint(low=0, high=source.shape[0],
                                        size=min([n_samples, source.shape[0]])), :]
               if n_samples is not None else source[:])
    samples = samples @ total_matrix[:3, :3].T + total_matrix[:3, -1]
    old_cost = np.inf
    for i in range(max_iterations):
        print('iteration', i, 'cost', old_cost)
        closest = target[tree.query(samples, 1)[1]]
        matrix, samples, cost = procrustes(samples, closest, scale=scale)
        total_matrix = np.dot(matrix, total_matrix)
        if old_cost - cost < threshold:
            break
        else:
            old_cost = cost
    aligned = source @ total_matrix[:3, :3].T + total_matrix[:3, -1]
    return total_matrix, aligned, cost


def combined_alignment(source, target, pre_align=True, scale=True, shear=False, iterations=100):
    """Align source to target by combination of moment-of-inertia based alignment + ICP."""
    if pre_align:
        trafo_initial, _ = align_by_centroid_and_inertia(source, target,
                                                         scale=scale, shear=shear,
                                                         improper=False)
    else:
        trafo_initial = np.eye(4)
    if iterations == 0:
        return trafo_initial
    trafo_icp, _, _ = icp(source, target, initial=trafo_initial,
                           threshold=1e-4, max_iterations=iterations,
                           scale=scale, n_samples=5000)
    return trafo_icp
