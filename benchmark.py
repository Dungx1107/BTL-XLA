import os
import cv2
import time
import numpy as np
import math
from typing import Dict, List, Callable


class FilterBenchmark:
    def __init__(self, input_dir: str = 'images', output_dir: str = 'results'):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.filters: Dict[str, Callable[[np.ndarray], np.ndarray]] = {}

    def register_filter(self, name: str, filter_func: Callable[[np.ndarray], np.ndarray]):
        """
        Đăng ký một phương pháp lọc mới vào hệ thống benchmark.
        filter_func: Hàm hoặc một phương thức nhận vào ảnh (np.ndarray) và trả về ảnh (np.ndarray)
        """
        self.filters[name] = filter_func

    def _calculate_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Tính PSNR giữa ảnh gốc và ảnh sau khi xử lý (chuẩn toán học)"""
        if img1.shape != img2.shape:
            raise ValueError("Hai ảnh phải có cùng kích thước để tính PSNR.")

        img1_float = img1.astype(np.float64)
        img2_float = img2.astype(np.float64)

        mse = np.mean((img1_float - img2_float) ** 2)
        if mse == 0:
            return float('inf')

        return 10 * math.log10((255.0 ** 2) / mse)

    def run(self, ground_truth_filter_name: str = None):
        """
        Chạy kiểm thử trên toàn bộ ảnh trong thư mục.
        ground_truth_filter_name: Tên của bộ lọc được chọn làm chuẩn độ chính xác (ví dụ: 'Standard')
                                  Nếu có, PSNR của các phương pháp khác sẽ được so với ảnh đầu ra của bộ lọc này.
                                  Nếu không có, PSNR sẽ được so với ảnh gốc (Original).
        """
        img_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not img_files:
            print(f"Không tìm thấy ảnh nào trong thư mục '{self.input_dir}'")
            return

        # Tạo Header hiển thị kết quả
        print("\n" + "=" * 110)
        header = f"{'IMAGE NAME':<20} | {'METHOD':<18} | {'TIME (s)':<12} | {'PSNR (dB)':<12} | {'STATUS':<15}"
        print(header)
        print("-" * 110)

        for f in img_files:
            img_path = os.path.join(self.input_dir, f)
            img = cv2.imread(img_path)
            if img is None:
                continue

            print(f"{f[:20]:<20}")

            # Thực thi tất cả các bộ lọc đã đăng ký để lấy kết quả ảnh đầu ra và thời gian
            outputs = {}
            times = {}

            for name, filter_func in self.filters.items():
                # Đo thời gian chạy thực tế
                start_time = time.time()
                try:
                    # Chạy hàm lọc
                    res_img = filter_func(img)
                    times[name] = time.time() - start_time
                    outputs[name] = res_img
                except Exception as e:
                    print(f"                     | {name:<18} | {'FAILED':<12} | {'--':<12} | {str(e)[:15]}")
                    continue

            # Xác định ảnh đích để so sánh chất lượng (Độ chính xác)
            # Theo bài báo, họ so sánh thuật toán tăng tốc O(1) với Standard Bilateral làm chuẩn (Ground Truth)
            if ground_truth_filter_name in outputs:
                base_img = outputs[ground_truth_filter_name]
                base_label = f"vs {ground_truth_filter_name}"
            else:
                base_img = img
                base_label = "vs Original"

            # Tính toán metric chất lượng ảnh và in kết quả chi tiết
            for name in outputs.keys():
                # Không tính toán chất lượng của chính bộ lọc làm chuẩn nếu so với chính nó
                if ground_truth_filter_name and name == ground_truth_filter_name:
                    psnr_val = float('inf')  # Tự đối chiếu với chính mình
                else:
                    psnr_val = self._calculate_psnr(base_img, outputs[name])

                psnr_str = f"{psnr_val:.4f}" if psnr_val != float('inf') else "INF"

                print(f"                     | {name:<18} | {times[name]:<12.4f} | {psnr_str:<12} | {base_label}")

                # Lưu ảnh kết quả ra thư mục kết quả để hậu kiểm trực quan
                out_name = f"{os.path.splitext(f)[0]}_{name}.png"
                cv2.imwrite(os.path.join(self.output_dir, out_name), outputs[name])

            print("-" * 110)