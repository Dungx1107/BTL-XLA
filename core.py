import numpy as np
import cv2
from scipy.spatial import KDTree


class O1BilateralFilter:
    def __init__(self, n_samples=20, sigma_s=15, sigma_r=0.1):
        self.n_samples = n_samples  # Số lượng mẫu màu [cite: 248, 252]
        self.sigma_s = sigma_s  # Bán kính lọc không gian [cite: 262]
        self.sigma_r = sigma_r  # Tham số dải (range parameter) [cite: 44, 186]

    def _adaptive_sampling(self, image):
        """Giai đoạn 1: Poisson disk sampling trong không gian màu [cite: 116, 119, 120]"""
        pixels = image.reshape(-1, 3)
        unique_colors = np.unique(pixels, axis=0)
        samples = []
        rs = 0.5
        while len(samples) < self.n_samples:
            idx = np.random.choice(len(unique_colors))
            color = unique_colors[idx]
            if not samples or np.all(np.linalg.norm(np.array(samples) - color, axis=1) > 2 * rs):
                samples.append(color)
            if len(samples) < self.n_samples and rs > 0.01:
                rs *= 0.95
        return np.array(samples)

    def apply(self, image):
        img_f = image.astype(np.float32) / 255.0
        h, w, c = img_f.shape
        sampled_colors = self._adaptive_sampling(img_f)

        j_stack, w_stack = [], []
        # Giai đoạn 2: Lọc O(1) [cite: 40, 81, 82]
        for k in sampled_colors:
            dist_sq = np.sum((img_f - k) ** 2, axis=2)
            gr = np.exp(-dist_sq / (2 * self.sigma_r ** 2))  # Công thức (2) [cite: 31]

            # Box filter có độ phức tạp O(1) [cite: 131]
            jk = cv2.boxFilter(img_f * gr[..., None], -1, (self.sigma_s, self.sigma_s))
            wk = cv2.boxFilter(gr, -1, (self.sigma_s, self.sigma_s))
            j_stack.append(jk)
            w_stack.append(wk)

        # Giai đoạn 3: Nội suy trọng số từ 4 màu gần nhất [cite: 155, 156]
        tree = KDTree(sampled_colors)
        dists, indices = tree.query(img_f, k=4)

        res_num = np.zeros_like(img_f)
        res_den = np.zeros((h, w, 1))
        d_min = dists[:, :, 0:1] + 1e-6

        for i in range(4):
            idx = indices[:, :, i]
            d = dists[:, :, i:i + 1]
            omega = np.exp(-d / (2 * d_min))  # Trọng số exponential [cite: 160]
            for s_idx in range(self.n_samples):
                mask = (idx == s_idx)
                if np.any(mask):
                    # Công thức (9) [cite: 157]
                    val = j_stack[s_idx][mask] / (w_stack[s_idx][mask][..., None] + 1e-8)
                    res_num[mask] += val * omega[mask]
                    res_den[mask] += omega[mask]

        return (np.clip(res_num / (res_den + 1e-8), 0, 1) * 255).astype(np.uint8)