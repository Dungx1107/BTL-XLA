import numpy as np
import cv2
from scipy.spatial import KDTree

class ConstantTimeBilateral:
    def __init__(self, n_samples=20, sigma_s=15, sigma_r=0.1):
        self.n_samples = n_samples
        self.sigma_s = sigma_s
        self.sigma_r = sigma_r

    def apply(self, image):
        img_f = image.astype(np.float32) / 255.0
        h, w, c = img_f.shape

        # Giai đoạn 1: Lấy mẫu Poisson Disk
        pixels = img_f.reshape(-1, 3)
        unique_colors = np.unique(pixels[::10], axis=0)  # Lấy mẫu thưa để nhanh
        samples = []
        rs = 0.5
        while len(samples) < self.n_samples:
            color = unique_colors[np.random.choice(len(unique_colors))]
            if not samples or np.all(np.linalg.norm(np.array(samples) - color, axis=1) > 2 * rs):
                samples.append(color)
            rs *= 0.98
            if rs < 0.001: break
        sampled_colors = np.array(samples)

        # Giai đoạn 2: Lọc O(1)
        j_stack, w_stack = [], []
        for k in sampled_colors:
            dist_sq = np.sum((img_f - k) ** 2, axis=2)
            gr = np.exp(-dist_sq / (2 * self.sigma_r ** 2))
            jk = cv2.boxFilter(img_f * gr[..., None], -1, (self.sigma_s, self.sigma_s))
            wk = cv2.boxFilter(gr, -1, (self.sigma_s, self.sigma_s))
            j_stack.append(jk)
            w_stack.append(wk)

        # Giai đoạn 3: Nội suy
        tree = KDTree(sampled_colors)
        dists, indices = tree.query(img_f, k=4)
        res_num, res_den = np.zeros_like(img_f), np.zeros((h, w, 1))
        d_min = dists[:, :, 0:1] + 1e-6
        for i in range(4):
            idx, d = indices[:, :, i], dists[:, :, i:i + 1]
            omega = np.exp(-d / (2 * d_min))
            for s_idx in range(len(sampled_colors)):
                mask = (idx == s_idx)
                if np.any(mask):
                    val = j_stack[s_idx][mask] / (w_stack[s_idx][mask][..., None] + 1e-8)
                    res_num[mask] += val * omega[mask]
                    res_den[mask] += omega[mask]
        return (np.clip(res_num / (res_den + 1e-8), 0, 1) * 255).astype(np.uint8)