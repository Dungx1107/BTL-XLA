from filters.constant_time_bilateral_2 import ConstantTimeBilateral
import numpy as np
import radix_sort_custom
import color_merge_cpp
import cv2
import os

def _variance(points):
    p = np.asarray(points, dtype=np.float32)
    mu = np.mean(p, axis=0)
    diff = p - mu
    return np.mean(np.sum(diff * diff, axis=1))

def _split(points, depth=0, var_threshold=500.0, max_depth=6, min_size=8):
    n = len(points)

    if n <= min_size or depth >= max_depth:
        return [np.mean(points, axis=0)]

    var = _variance(points)
    if var < var_threshold:
        return [np.mean(points, axis=0)]

    # octree split (bit-based)
    bit = 7 - depth

    octants = (
        (((points[:, 0] >> bit) & 1) << 2) |
        (((points[:, 1] >> bit) & 1) << 1) |
        ((points[:, 2] >> bit) & 1)
    )

    result = []
    for oid in range(8):
        mask = octants == oid
        if np.any(mask):
            result.extend(
                _split(points[mask], depth + 1, var_threshold, max_depth, min_size)
            )

    return result

class VarianceSplitVer(ConstantTimeBilateral):
    def __init__(self, tile_size=128, sigma_s=15, sigma_r=0.1):
        super().__init__(None, sigma_s, sigma_r)
        self.tile_size = tile_size
        self.pad = int(np.ceil(self.sigma_s / 2))

    def _poisson_disk_sampling(self, img_f):
        pixels = (img_f.reshape(-1, 3) * 255).astype(np.uint8)

        palette = _split(
            pixels,
            depth=0,
            var_threshold=200,
            max_depth=6,
            min_size=8
        )

        palette = np.asarray(np.round(palette), dtype=np.uint8)

        palette = color_merge_cpp.merge(palette, 50.0)

        return palette.astype(np.float32) / 255.0
    
    def _select_tile(self, img, i, x1, j, y1):
        return img[i:x1, j:y1]

    def apply(self, image):
        h, w, c = image.shape
        out = np.zeros((h, w, c), dtype=np.float32)

        tile_h = self.tile_size
        tile_w = self.tile_size

        n_rows = (h + tile_h - 1) // tile_h
        n_cols = (w + tile_w - 1) // tile_w

        for ti in range(n_rows):
            for tj in range(n_cols):

                i = ti * tile_h
                j = tj * tile_w
                x1 = min(i + tile_h, h)
                y1 = min(j + tile_w, w)

                th = x1 - i
                tw = y1 - j

                patch = self._select_tile(image, i, x1, j, y1)

                processed = super().apply(patch.astype(np.uint8))

                processed = processed.astype(np.float32)

                if processed.shape[0] != th or processed.shape[1] != tw:
                    processed = cv2.resize(
                        processed,
                        (tw, th),
                        interpolation=cv2.INTER_LINEAR
                    )

                out[i:x1, j:y1] = processed / 255.0

        return (np.clip(out, 0, 1) * 255).astype(np.uint8)