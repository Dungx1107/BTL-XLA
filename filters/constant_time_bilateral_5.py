from .constant_time_bilateral_3 import ConstantTimeBilateralStrict
import numpy as np
import radix_sort_custom


def _variance(points):
    p = np.asarray(points, dtype=np.float32)
    mu = np.mean(p, axis=0)
    diff = p - mu
    return np.mean(np.sum(diff * diff, axis=1))

def _split(points, dim=0, var_threshold=500.0, min_size=32):
    points = np.asarray(points, dtype=np.uint8)

    n = len(points)

    if n <= min_size:
        return [np.mean(points, axis=0)]

    var = _variance(points)

    if var < var_threshold:
        return [np.mean(points, axis=0)]

    points = radix_sort_custom.sort_by_dim(points, dim)

    mid = n // 2

    if mid < min_size:
        return [np.mean(points, axis=0)]

    next_dim = (dim + 1) % 3

    return (
        _split(points[:mid], next_dim, var_threshold, min_size) +
        _split(points[mid:], next_dim, var_threshold, min_size)
    )


class VarianceSplitVer(ConstantTimeBilateralStrict):
    def __init__(self, sigma_s=15, sigma_r=0.1):
        super().__init__(None, sigma_s, sigma_r)

    def _poisson_disk_sampling(self, img_f):
        pixels = img_f.reshape(-1, 3).astype(np.uint8)
        palette = _split(pixels)
        return np.asarray(palette, dtype=np.float32) / 255.0

    def apply(self, image):
        img_f = image.astype(np.uint8)
        sampled_colors = self._poisson_disk_sampling(img_f)
        j_stack, w_stack = self._spatial_filter_stacks_vectorized(img_f, sampled_colors)
        res_f = self._interpolate_results_strict(img_f, sampled_colors, j_stack, w_stack)

        return (np.clip(res_f, 0, 1) * 255).astype(np.uint8)