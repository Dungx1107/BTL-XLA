import numpy as np
import cv2
from scipy.spatial import KDTree


class ConstantTimeBilateralStrict:
    def __init__(self, n_samples=20, sigma_s=15, sigma_r=0.1):
        self.n_samples = n_samples
        self.sigma_s = sigma_s
        # sigma_r tương ứng với thang đo chuẩn hóa của bài báo [cite: 197]
        self.sigma_r = sigma_r

    def _poisson_disk_sampling(self, img_f):
        """
        Giai đoạn 1: Adaptive Sampling tuân thủ Mục 3.2 [cite: 115]
        Lấy toàn bộ màu của ảnh làm pool đại diện.
        """
        pixels = img_f.reshape(-1, 3)
        # Loại bỏ các màu trùng lặp chính xác để thu gọn pool ban đầu không mất thông tin
        unique_colors = np.unique(pixels, axis=0)

        samples = []
        # Bán kính khởi tạo trong không gian màu [0, 1]^3
        rs = 0.5

        # Vòng lặp giảm bán kính rs khi không tìm thêm được không gian
        while len(samples) < self.n_samples and rs > 0.001:
            # Lựa chọn ngẫu nhiên từ pool màu thực tế của ảnh [cite: 122]
            idx = np.random.choice(len(unique_colors))
            color = unique_colors[idx]

            if not samples or np.all(np.linalg.norm(np.array(samples) - color, axis=1) > 2 * rs):
                samples.append(color)
            else:
                # Giảm bán kính thích ứng theo mô tả của bài báo [cite: 123]
                rs *= 0.95

                # Nếu vẫn thiếu mẫu, bổ sung trực tiếp từ phân phối thực tế
        while len(samples) < self.n_samples:
            idx = np.random.choice(len(unique_colors))
            samples.append(unique_colors[idx])

        return np.array(samples[:self.n_samples])

    def _spatial_filter_stacks_vectorized(self, img_f, sampled_colors):
        """
        Giai đoạn 2: Lọc không gian O(1) tuân thủ Phương trình (2) & (6) [cite: 31, 78, 130]
        """
        h, w, c = img_f.shape
        n_k = len(sampled_colors)

        # Tính tổng bình phương khoảng cách sai lệch màu theo Eq (2)
        dist_sq = np.sum((img_f[:, :, None, :] - sampled_colors[None, None,:, :]) ** 2, axis=3)
        gr_stack = np.exp(-dist_sq / (2 * (self.sigma_r ** 2)))  # Shape: (h, w, n_k)

        j_stack = np.zeros((h, w, n_k, c), dtype=np.float32)
        w_stack = np.zeros((h, w, n_k), dtype=np.float32)

        # Áp dụng bộ lọc hộp O(1) (Box Filter) qua ảnh tích hợp [cite: 81, 131]
        for idx in range(n_k):
            gr = gr_stack[:, :, idx]
            # Kiếm tra tích g_r(k, T) * I theo công thức (6) [cite: 78, 80]
            j_stack[:, :, idx, :]= cv2.boxFilter(img_f * gr[..., None], -1, (self.sigma_s, self.sigma_s),
                                                 borderType=cv2.BORDER_REFLECT)
            w_stack[:, :, idx] = cv2.boxFilter(gr, -1, (self.sigma_s, self.sigma_s), borderType=cv2.BORDER_REFLECT)

        return j_stack, w_stack

    def _interpolate_results_strict(self, img_f, sampled_colors, j_stack, w_stack):
        """
        Giai đoạn 3: Nội suy tuân thủ cấu trúc Phương trình (9)
        """
        h, w, c = img_f.shape

        # Khởi tạo KDTree toán học tính toán khoảng cách L2 chính xác (p=2)
        tree = KDTree(sampled_colors)
        dists, indices = tree.query(img_f, k=4, p=2)  # dists: Khoảng cách d thực tế (không phải d^2)

        # Trích xuất d_min để tính toán trọng số mũ chuẩn xác theo Mục 3.4
        d_min = dists[:, :, 0:1] + 1e-8
        omega = np.exp(-dists / (2 * d_min))  # Shape: (h, w, 4)

        # Trích xuất thành phần J và W của 4 lân cận gần nhất bằng Advanced Indexing
        # Tránh việc thực hiện chia vô hướng trên toàn bộ không gian stack mẫu
        grid_h, grid_w = np.arange(h)[:, None, None], np.arange(w)[None, :, None]

        j_chosen = j_stack[grid_h, grid_w, indices]  # Shape: (h, w, 4, c)
        w_chosen = w_stack[grid_h, grid_w, indices]  # Shape: (h, w, 4)

        # Tính toán giá trị J^k / W^k của từng thành phần tương ứng trước khi nội suy
        component_divided = j_chosen / (w_chosen[..., None] + 1e-8)

        # Tiến hành nhân trọng số omega_i tuân thủ nghiêm ngặt phương trình (9)
        res_num = np.sum(component_divided * omega[..., None], axis=2)
        res_den = np.sum(omega, axis=2, keepdims=True)

        return res_num / (res_den + 1e-8)

    def apply(self, image):
        img_f = image.astype(np.float32) / 255.0

        sampled_colors = self._poisson_disk_sampling(img_f)
        j_stack, w_stack = self._spatial_filter_stacks_vectorized(img_f, sampled_colors)
        res_f = self._interpolate_results_strict(img_f, sampled_colors, j_stack, w_stack)

        return (np.clip(res_f, 0, 1) * 255).astype(np.uint8)