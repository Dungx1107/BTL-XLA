from filters.constant_time_bilateral_2 import ConstantTimeBilateral
import numpy as np
import radix_sort_custom
import color_merge_cpp

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

class BoxSplitVer(ConstantTimeBilateral):
    def __init__(self, color_dis_threshold=10.0, sigma_s=15, sigma_r=0.1):
        super().__init__(None, sigma_s, sigma_r)
        self.color_dis_threshold = color_dis_threshold

    def _poisson_disk_sampling(self, img_f):

        pixels = (img_f.reshape(-1, 3) * 255).astype(np.uint8)

        palette = _split(pixels)

        palette = np.asarray(
            np.round(palette),
            dtype=np.uint8
        )

        palette = color_merge_cpp.merge(
            palette,
            self.color_dis_threshold
        )

        return palette.astype(np.float32) / 255.0
        