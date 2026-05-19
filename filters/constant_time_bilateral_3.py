import numpy as np
import cv2
from scipy.spatial import KDTree


class ConstantTimeBilateral:
    def __init__(self, n_samples=20, sigma_s=15, sigma_r=0.1):
        self.n_samples = n_samples
        self.sigma_s = sigma_s
        self.sigma_r = sigma_r

    def _poisson_disk_sampling(self, img_f):
        """Giai đoạn 1: Giữ nguyên (do tính chất tuần tự của Poisson Disk)."""
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

    def _spatial_filter_stacks_vectorized(self, img_f, sampled_colors):
        """Giai đoạn 2: Tính toán song song các khoảng cách màu bằng Broadcasting."""
        h, w, c = img_f.shape
        n_k = len(sampled_colors)

        # Broadcasting để tính khoảng cách của toàn bộ pixel tới toàn bộ màu mẫu cùng lúc
        # img_f[:, :, None, :] -> (h, w, 1, 3)
        # sampled_colors[None, None, :, :] -> (1, 1, n_k, 3)
        # dist_sq có shape: (h, w, n_k)
        dist_sq = np.sum((img_f[:, :, None, :] - sampled_colors[None, None, :, :]) ** 2, axis=3)
        gr_stack = np.exp(-dist_sq / (2 * self.sigma_r ** 2))  # Shape: (h, w, n_k)

        # Tách biệt BoxFilter: OpenCV không hỗ trợ ảnh nhiều hơn 4 kênh trực tiếp,
        # nhưng ta có thể tận dụng cấu trúc mảng để tính toán nhanh.
        j_stack = np.zeros((h, w, n_k, c), dtype=np.float32)
        w_stack = np.zeros((h, w, n_k), dtype=np.float32)

        for idx in range(n_k):
            gr = gr_stack[:, :, idx]
            j_stack[:, :, idx, :] = cv2.boxFilter(img_f * gr[..., None], -1, (self.sigma_s, self.sigma_s))
            w_stack[:, :, idx] = cv2.boxFilter(gr, -1, (self.sigma_s, self.sigma_s))

        return j_stack, w_stack

    def _interpolate_results_vectorized(self, img_f, sampled_colors, j_stack, w_stack):
        """Giai đoạn 3: Vector hóa hoàn toàn, loại bỏ vòng lặp qua từng màu mẫu."""
        h, w, c = img_f.shape
        n_k = len(sampled_colors)

        # KDTree query giữ nguyên (đã tối ưu bằng C-extension phía sau)
        tree = KDTree(sampled_colors)
        dists, indices = tree.query(img_f, k=4)  # Cả hai đều có shape: (h, w, 4)

        d_min = dists[:, :, 0:1] + 1e-6
        omega = np.exp(-dists / (2 * d_min))  # Shape: (h, w, 4)

        # Tạo mask nhị phân dạng One-hot để chọn nhanh giá trị từ stack mà không cần loop
        # indices[:, :, :, None] -> (h, w, 4, 1)
        # np.arange(n_k) -> (n_k,)
        # mask có shape: (h, w, 4, n_k)
        mask = (indices[:, :, :, None] == np.arange(n_k)[None, None, None, :])

        # Chuẩn hóa stack trước khi nội suy: (h, w, n_k, c)
        val_stack = j_stack / (w_stack[..., None] + 1e-8)

        # Trích xuất các giá trị tương ứng với indices thông qua phép nhân ma trận (Einsum) hoặc chọn lọc nâng cao
        # Để tránh tốn bộ nhớ quá mức, dùng nâng cao mảng trỏ (Advanced Indexing)
        # Lấy ra các giá trị tương ứng với K lân cận gần nhất
        # j_chosen shape: (h, w, 4, c), w_chosen shape: (h, w, 4)
        j_chosen = val_stack[np.arange(h)[:, None, None], np.arange(w)[None, :, None], indices]

        # Tính toán kết quả nội suy cuối cùng bằng nhân vô hướng vector hóa chéo qua trục K lân cận (trục 2)
        # omega[..., None] có shape (h, w, 4, 1)
        res_num = np.sum(j_chosen * omega[..., None], axis=2)  # Shape: (h, w, c)
        res_den = np.sum(omega, axis=2, keepdims=True)  # Shape: (h, w, 1)

        return res_num / (res_den + 1e-8)

    def apply(self, image):
        img_f = image.astype(np.float32) / 255.0

        sampled_colors = self._poisson_disk_sampling(img_f)
        j_stack, w_stack = self._spatial_filter_stacks_vectorized(img_f, sampled_colors)
        res_f = self._interpolate_results_vectorized(img_f, sampled_colors, j_stack, w_stack)

        return (np.clip(res_f, 0, 1) * 255).astype(np.uint8)