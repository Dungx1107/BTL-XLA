from .constant_time_bilateral_3 import ConstantTimeBilateralStrict
import numpy as np
import radix_sort_custom

def _split(points, dim=0):
    points = radix_sort_custom.sort_by_dim(
        np.asarray(points, dtype=np.uint8),
        dim
    )

    n = len(points)
    q1 = n // 3
    q2 = 2 * n // 3

    if dim == 2:
        def centroid(chunk):
            return np.mean(chunk, axis=0)

        return [
            centroid(points[:q1]),
            centroid(points[q1:q2]),
            centroid(points[q2:])
        ]

    return (
        _split(points[:q1], dim + 1) +
        _split(points[q1:q2], dim + 1) +
        _split(points[q2:], dim + 1)
    )

class BoxSplitVer(ConstantTimeBilateralStrict):
    def __init__(self, sigma_s=15, sigma_r=0.1):
        super().__init__(None, sigma_s, sigma_r)

    def _poisson_disk_sampling(self, img_f):
        pixels = img_f.reshape(-1, 3).astype(np.uint8)
        palette = _split(pixels)
        return np.asarray(palette, dtype=np.float32) / 255
    
    def apply(self, image):
        img_f = image.astype(np.uint8)
        sampled_colors = self._poisson_disk_sampling(img_f)
        j_stack, w_stack = self._spatial_filter_stacks_vectorized(img_f, sampled_colors)
        res_f = self._interpolate_results_strict(img_f, sampled_colors, j_stack, w_stack)

        return (np.clip(res_f, 0, 1) * 255).astype(np.uint8)