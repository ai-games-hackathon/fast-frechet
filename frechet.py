import math
import numpy as np
from numba import jit


@jit('float64(float64[:], float64[:])', nopython=True, fastmath=True)
def _euclidean(p: np.ndarray, q: np.ndarray) -> float:
    d = p - q
    return math.sqrt(np.dot(d, d))

@jit('float64(float64[:,:], int64, int64)', nopython=True)
def _get_corner_min_array(f_mat: np.ndarray, i: int, j: int) -> float:
    if i > 0 and j > 0:
        a = min(f_mat[i - 1, j - 1],
                f_mat[i, j - 1],
                f_mat[i - 1, j])
    elif i == 0 and j == 0:
        a = f_mat[i, j]
    elif i == 0:
        a = f_mat[i, j - 1]
    else:  # j == 0:
        a = f_mat[i - 1, j]
    return a

@jit('float64[:,:](float64[:,:], int64[:, :], float64[:, :], float64[:, :])', nopython=True)
def _fast_frechet_matrix(dist: np.ndarray, diag: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    for k in range(diag.shape[0]):
        i0 = diag[k, 0]
        j0 = diag[k, 1]

        for i in range(i0, p.shape[0]):
            if np.isfinite(dist[i, j0]):
                c = _get_corner_min_array(dist, i, j0)
                if c > dist[i, j0]:
                    dist[i, j0] = c
            else:
                break

        # add 1 to j0 to avoid recalculating the diagonal
        for j in range(j0 + 1, q.shape[0]):
            if np.isfinite(dist[i0, j]):
                c = _get_corner_min_array(dist, i0, j)
                if c > dist[i0, j]:
                    dist[i0, j] = c
            else:
                break
    return dist

@jit('float64[:,:](float64[:, :], float64[:, :], int64[:, :])', nopython=True)
def _fast_distance_matrix(p: np.ndarray, q: np.ndarray, diag: np.ndarray) -> np.ndarray:
    n_diag = diag.shape[0]
    diag_max = 0.0
    i_min = 0
    j_min = 0
    p_count = p.shape[0]
    q_count = q.shape[0]

    # create the distance array
    dist = np.full((p_count, q_count), np.inf, dtype=np.float64)

    # fill in the diagonal with the seed distance values
    for k in range(n_diag):
        i0 = diag[k, 0]
        j0 = diag[k, 1]
        d = _euclidean(p[i0], q[j0])
        diag_max = max(diag_max, d)
        dist[i0, j0] = d

    for k in range(n_diag - 1):
        i0 = diag[k, 0]
        j0 = diag[k, 1]
        p_i0 = p[i0]
        q_j0 = q[j0]

        for i in range(i0 + 1, p_count):
            if np.isinf(dist[i, j0]):
                d = _euclidean(p[i], q_j0)
                if d < diag_max or i < i_min:
                    dist[i, j0] = d
                else:
                    break
            else:
                break
        i_min = i

        for j in range(j0 + 1, q_count):
            if np.isinf(dist[i0, j]):
                d = _euclidean(p_i0, q[j])
                if d < diag_max or j < j_min:
                    dist[i0, j] = d
                else:
                    break
            else:
                break
        j_min = j
    return dist

@jit('int64[:, :](int64, int64, int64, int64)', nopython=True)
def _bresenham_pairs(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dim = max(dx, dy)
    pairs = np.zeros((dim, 2), dtype=np.int64)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx // 2
        for i in range(dx):
            pairs[i, 0] = x
            pairs[i, 1] = y
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy // 2
        for i in range(dy):
            pairs[i, 0] = x
            pairs[i, 1] = y
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    return pairs

@jit('float64[:, :](float64[:, :], float64[:, :])', nopython=True)
def _fdfd_matrix(p: np.ndarray, q: np.ndarray) -> float:
    diagonal = _bresenham_pairs(0, 0, p.shape[0], q.shape[0])
    ca = _fast_distance_matrix(p, q, diagonal)
    ca = _fast_frechet_matrix(ca, diagonal, p, q)
    return ca

def frechet_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute fast discrete Frechet distance between trajectories p and q
    :param p: trajectory p with shape (P, 2)
    :param q: trajectory q with shape (Q, 2)
    :returns: discrete Frechet distance
    """
    ca = _fdfd_matrix(p, q)
    return ca[p.shape[0]-1, q.shape[0]-1]
