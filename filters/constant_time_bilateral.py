"""
Constant Time Bilateral Filtering for Color Images
Triển khai chuẩn theo bài báo:
  Tu et al., "Constant Time Bilateral Filtering for Color Images", ICIP 2016

Ba giai đoạn theo Mục 3:
  Stage 1 — Adaptive Poisson Disk Sampling (Mục 3.2)
  Stage 2 — O(1) Spatial Filtering (Mục 3.3)
  Stage 3 — Interpolation theo Eq. 9 (Mục 3.4)
"""

import numpy as np
import cv2
from scipy.spatial import KDTree


class ConstantTimeBilateral:
    """
    Tham số
    -------
    n_samples : int
        Số màu mẫu (bài báo: 20 mẫu là đủ cho chất lượng tốt, Mục 4.1)
    filter_radius : int
        Bán kính lọc không gian r — kích thước cửa sổ = 2*r+1 (Mục 3.3)
    sigma_r : float
        Tham số range kernel (chuẩn hóa về [0,1]), Eq. 2
    spatial_filter : str
        'box' hoặc 'gaussian' — bài báo hỗ trợ cả hai (Mục 3.3)
    """

    def __init__(
        self,
        n_samples: int = 20,
        filter_radius: int = 15,
        sigma_r: float = 0.1,
        spatial_filter: str = "box",
    ):
        self.n_samples = n_samples
        self.filter_radius = filter_radius
        self.sigma_r = sigma_r
        self.spatial_filter = spatial_filter

    # ------------------------------------------------------------------
    # Stage 1: Adaptive Poisson Disk Sampling (Mục 3.2)
    # ------------------------------------------------------------------
    def _poisson_disk_sampling(self, img_f: np.ndarray) -> np.ndarray:
        """
        Lấy mẫu màu thích ứng từ phân phối màu thực của ảnh.

        Thuật toán đúng theo bài báo:
        - Lấy TOÀN BỘ màu ảnh làm pool (không lấy thưa).
        - Thử nhiều ứng viên ngẫu nhiên từ pool trước khi giảm r_s.
        - Chỉ giảm r_s khi đã thử MAX_ATTEMPTS lần mà không tìm được
          ứng viên hợp lệ nào (pool bão hòa tại r_s hiện tại).
        - Số mẫu không bị giới hạn bởi N^3 như [7] (ưu điểm bài báo).
        """
        # Lấy toàn bộ màu duy nhất của ảnh làm pool
        pixels = img_f.reshape(-1, 3)
        unique_colors = np.unique(pixels, axis=0)  # (M, 3)

        samples = []
        rs = 0.5          # Bán kính đĩa Poisson khởi đầu
        MAX_ATTEMPTS = 30  # Số lần thử tối đa trước khi giảm r_s

        while len(samples) < self.n_samples and rs > 1e-4:
            accepted = False
            for _ in range(MAX_ATTEMPTS):
                idx = np.random.randint(len(unique_colors))
                candidate = unique_colors[idx]

                # Chấp nhận nếu cách mọi mẫu đã có > 2*r_s (Mục 3.2)
                if len(samples) == 0:
                    samples.append(candidate)
                    accepted = True
                    break
                dists = np.linalg.norm(
                    np.array(samples, dtype=np.float32) - candidate, axis=1
                )
                if np.all(dists > 2 * rs):
                    samples.append(candidate)
                    accepted = True
                    break

            # Chỉ giảm r_s khi pool đã bão hòa (không tìm được mẫu mới)
            if not accepted:
                rs *= 0.9

        # Fallback an toàn: nếu ảnh có quá ít màu duy nhất
        # → lặp lại các màu có sẵn để đủ n_samples
        if len(samples) < self.n_samples:
            rng = np.random.default_rng(0)
            extra_idx = rng.choice(
                len(unique_colors),
                size=self.n_samples - len(samples),
                replace=True,
            )
            for i in extra_idx:
                samples.append(unique_colors[i])

        return np.array(samples[: self.n_samples], dtype=np.float32)  # (n_samples, 3)

    # ------------------------------------------------------------------
    # Stage 2: O(1) Spatial Filtering (Mục 3.3, Eq. 3 & 4)
    # ------------------------------------------------------------------
    def _spatial_filter(self, src: np.ndarray) -> np.ndarray:
        """
        Lọc không gian O(1).

        Bài báo hỗ trợ hai loại (Mục 3.3):
          - Box filter   : O(1) qua integral image
          - Gaussian IIR : O(1) qua IIR recursive filter

        Kernel size đúng: (2*r+1) × (2*r+1) với r = filter_radius.
        KHÔNG dùng sigma_s làm kích thước kernel.
        """
        ksize = 2 * self.filter_radius + 1

        if self.spatial_filter == "box":
            # cv2.boxFilter với normalize=True ≡ box filter chuẩn
            return cv2.boxFilter(
                src, ddepth=-1, ksize=(ksize, ksize),
                borderType=cv2.BORDER_REFLECT,
            )
        else:
            # Gaussian IIR — sigma_s là độ lệch chuẩn không gian
            sigma_s = self.filter_radius / 3.0  # quy ước: r ≈ 3*sigma
            return cv2.GaussianBlur(
                src, ksize=(0, 0), sigmaX=sigma_s,
                borderType=cv2.BORDER_REFLECT,
            )

    def _compute_filter_stacks(
        self, img_f: np.ndarray, sampled_colors: np.ndarray
    ):
        """
        Tính J_k và W_k cho mọi màu mẫu k (Eq. 3 & 4).

          J_k(p) = Σ_q  g_s(p,q) · g_r(k, I(q)) · I(q)   [Eq. 3]
          W_k(p) = Σ_q  g_s(p,q) · g_r(k, I(q))            [Eq. 4]

        Với g_r(k, I(q)) = exp( -||k - I(q)||² / 2σ_r² )   [Eq. 2]

        Vectorized hoàn toàn trên trục mẫu để tránh Python loop
        khi tính range kernel; chỉ loop khi gọi spatial filter
        (không thể vector hóa cv2 trên batch).
        """
        h, w, c = img_f.shape
        n_k = len(sampled_colors)

        # Range kernel: (h, w, n_k)
        # ||I(q) - k||² với k là từng màu mẫu
        diff = img_f[:, :, None, :] - sampled_colors[None, None, :, :]  # (h,w,n_k,3)
        dist_sq = np.sum(diff ** 2, axis=3)                              # (h,w,n_k)
        gr_stack = np.exp(-dist_sq / (2.0 * self.sigma_r ** 2))          # (h,w,n_k)

        j_stack = np.empty((h, w, n_k, c), dtype=np.float32)
        w_stack = np.empty((h, w, n_k),    dtype=np.float32)

        for idx in range(n_k):
            gr = gr_stack[:, :, idx].astype(np.float32)          # (h,w)
            # J_k = g_s * (g_r · I)  — lọc không gian trên ảnh đã nhân g_r
            j_stack[:, :, idx, :] = self._spatial_filter(
                (img_f * gr[:, :, None]).astype(np.float32)
            )
            # W_k = g_s * g_r        — lọc không gian trên g_r đơn thuần
            w_stack[:, :, idx] = self._spatial_filter(gr)

        return j_stack, w_stack  # (h,w,n_k,c), (h,w,n_k)

    # ------------------------------------------------------------------
    # Stage 3: Interpolation — Eq. 9 (Mục 3.4)
    # ------------------------------------------------------------------
    def _interpolate(
        self,
        img_f: np.ndarray,
        sampled_colors: np.ndarray,
        j_stack: np.ndarray,
        w_stack: np.ndarray,
    ) -> np.ndarray:
        """
        Nội suy kết quả lọc theo Eq. 9:

          I_F(p) = Σ_i  ω_i · [J_{k_i}(p) / W_{k_i}(p)]
                   ─────────────────────────────────────────
                              Σ_i  ω_i

        trong đó:
          - k_1..k_4  : 4 màu mẫu gần nhất với I(p) trong không gian màu
          - d_i       : khoảng cách Euclidean từ I(p) đến k_i
          - d_min     : min(d_1, d_2, d_3, d_4)
          - ω_i       = exp( −d_i / (2 · d_min) )     ← ĐÚNG theo bài báo

        Ghi chú quan trọng về ω_i:
          Bài báo viết: ω_i = exp(−d_i / 2·d_min)
          → dùng d (khoảng cách thực), KHÔNG dùng d² hay d_min².
        """
        h, w, c = img_f.shape

        # Tìm 4 màu mẫu gần nhất qua KDTree (khoảng cách Euclidean, p=2)
        tree = KDTree(sampled_colors)
        dists, indices = tree.query(img_f.reshape(-1, 3), k=4, p=2)
        # dists, indices: (h*w, 4)

        dists   = dists.reshape(h, w, 4).astype(np.float32)    # (h,w,4)
        indices = indices.reshape(h, w, 4)                      # (h,w,4)

        # Trọng số ω_i theo đúng bài báo: exp(−d_i / 2·d_min)
        d_min = dists[:, :, 0:1] + 1e-8                         # (h,w,1)
        omega = np.exp(-dists / (2.0 * d_min))                  # (h,w,4)

        # Advanced indexing: lấy J_{k_i} và W_{k_i} cho mọi pixel cùng lúc
        gi = np.arange(h)[:, None, None]   # (h,1,1)
        gj = np.arange(w)[None, :, None]   # (1,w,1)

        j_sel = j_stack[gi, gj, indices]   # (h,w,4,c)
        w_sel = w_stack[gi, gj, indices]   # (h,w,4)

        # Tính J_k / W_k cho từng trong 4 thành phần — Eq. 9
        # w_sel thêm chiều cuối để broadcast với j_sel
        component = j_sel / (w_sel[:, :, :, None] + 1e-8)      # (h,w,4,c)

        # Tổng có trọng số: Σ ω_i · (J_ki / W_ki)
        numerator   = np.sum(component * omega[:, :, :, None], axis=2)  # (h,w,c)
        denominator = np.sum(omega, axis=2, keepdims=True)               # (h,w,1)

        return numerator / (denominator + 1e-8)                          # (h,w,c)

    # ------------------------------------------------------------------
    # Hàm chính
    # ------------------------------------------------------------------
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Áp dụng O(1) bilateral filter lên ảnh màu BGR (uint8).

        Trả về ảnh đã lọc, cùng dtype và shape với input.
        """
        img_f = image.astype(np.float32) / 255.0  # chuẩn hóa về [0,1]

        # Stage 1
        sampled_colors = self._poisson_disk_sampling(img_f)

        # Stage 2
        j_stack, w_stack = self._compute_filter_stacks(img_f, sampled_colors)

        # Stage 3
        result_f = self._interpolate(img_f, sampled_colors, j_stack, w_stack)

        return (np.clip(result_f, 0.0, 1.0) * 255).astype(np.uint8)

