import numpy as np
import cv2
from scipy.spatial import KDTree


class ConstantTimeBilateral:
    def __init__(self, n_samples=20, sigma_s=15, sigma_r=0.1):
        self.n_samples = n_samples
        self.sigma_s = sigma_s
        self.sigma_r = sigma_r

    def _poisson_disk_sampling(self, img_f):
        """Giai đoạn 1: Lấy mẫu không gian màu dựa trên phân phối Poisson Disk."""
        pixels = img_f.reshape(-1, 3)
        unique_colors = np.unique(pixels[::10], axis=0)

        samples = []
        rs = 0.5
        while len(samples) < self.n_samples:
            color = unique_colors[np.random.choice(len(unique_colors))]
            if not samples or np.all(np.linalg.norm(np.array(samples) - color, axis=1) > 2 * rs):
                samples.append(color)
            rs *= 0.98
            if rs < 0.001:
                break
        return np.array(samples)

    def _spatial_filter_stacks(self, img_f, sampled_colors):
        """Giai đoạn 2: Tính toán các lớp ảnh (stacks) đã được lọc không gian O(1) qua Box Filter."""
        j_stack = []
        w_stack = []

        for k in sampled_colors:
            # Khoảng cách cường độ màu (Range distance)
            dist_sq = np.sum((img_f - k) ** 2, axis=2)
            gr = np.exp(-dist_sq / (2 * self.sigma_r ** 2))

            # Lọc không gian O(1) bằng Box Filter
            jk = cv2.boxFilter(img_f * gr[..., None], -1, (self.sigma_s, self.sigma_s))
            wk = cv2.boxFilter(gr, -1, (self.sigma_s, self.sigma_s))

            j_stack.append(jk)
            w_stack.append(wk)

        return j_stack, w_stack

    def _interpolate_results(self, img_f, sampled_colors, j_stack, w_stack):
        """Giai đoạn 3: Nội suy kết quả dựa trên khoảng cách KD-Tree của các màu đã lấy mẫu."""
        h, w, _ = img_f.shape
        tree = KDTree(sampled_colors)
        dists, indices = tree.query(img_f, k=4)

        res_num = np.zeros_like(img_f)
        res_den = np.zeros((h, w, 1))
        d_min = dists[:, :, 0:1] + 1e-6

        for i in range(4):
            idx = indices[:, :, i]
            d = dists[:, :, i:i + 1]
            omega = np.exp(-d / (2 * d_min))

            for s_idx in range(len(sampled_colors)):
                mask = (idx == s_idx)
                if np.any(mask):
                    # Chuẩn hóa giá trị từ stack
                    val = j_stack[s_idx][mask] / (w_stack[s_idx][mask][..., None] + 1e-8)
                    res_num[mask] += val * omega[mask]
                    res_den[mask] += omega[mask]

        return res_num / (res_den + 1e-8)

    def apply(self, image):
        """Hàm điều phối luồng xử lý chính."""
        # Chuẩn hóa dữ liệu đầu vào
        img_f = image.astype(np.float32) / 255.0

        # Thực hiện các bước độc lập
        sampled_colors = self._poisson_disk_sampling(img_f)
        j_stack, w_stack = self._spatial_filter_stacks(img_f, sampled_colors)
        res_f = self._interpolate_results(img_f, sampled_colors, j_stack, w_stack)

        # Chuyển đổi về định dạng ảnh gốc
        return (np.clip(res_f, 0, 1) * 255).astype(np.uint8)