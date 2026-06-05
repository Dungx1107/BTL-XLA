import cv2
import numpy as np
import time
import os
import csv
from datetime import datetime
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Import chính xác cả 2 phiên bản thuật toán để đối chiếu
from filters.constant_time_bilateral import ConstantTimeBilateral
from filters.constant_time_bilateral_2 import ConstantTimeBilateralStrict
from filters.standard_bilateral import apply_standard_bilateral
from filters.gaussian_blur import apply_gaussian_blur


def main():
    input_dir = 'images'
    output_dir = 'results_comparison'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'csv'), exist_ok=True)

    # Cấu hình tham số bộ lọc
    S_S = 35
    S_R = 0.08
    N_SAMPLES = 20

    img_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Khởi tạo tách biệt 2 thực thể thuật toán
    ct_base = ConstantTimeBilateral(n_samples=N_SAMPLES, sigma_s=S_S, sigma_r=S_R)
    ct_strict = ConstantTimeBilateralStrict(n_samples=N_SAMPLES, sigma_s=S_S, sigma_r=S_R)

    # --- CẤU HÌNH FILE CSV SO SÁNH HAI PHIÊN BẢN O(1) ---
    csv_file_path = os.path.join(output_dir, 'csv/complete_benchmark.csv')
    csv_headers = [
        "Timestamp", "Image_Name", "Sigma_S", "Sigma_R",
        "Gaussian_Time (s)", "Standard_Time (s)",
        "O1_Base_Time (s)", "O1_Strict_Time (s)",
        "O1_Base_PSNR (dB)", "O1_Strict_PSNR (dB)",
        "O1_Base_SSIM", "O1_Strict_SSIM"
    ]

    file_exists = os.path.isfile(csv_file_path)
    with open(csv_file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(csv_headers)
    # -----------------------------------------------------

    # Header hiển thị cho Console hệ thống
    print("\n" + "=" * 160)
    print(f"{'IMAGE NAME':<18} | {'GAUSSIAN':<10} | {'STANDARD':<10} | "
          f"{'O(1) BASE TIME':<15} | {'O(1) STRICT TIME':<17} | "
          f"{'PSNR BASE':<11} | {'PSNR STRICT':<13} | {'SSIM STRICT':<11}")
    print("-" * 160)

    for f_name in img_files:
        img = cv2.imread(os.path.join(input_dir, f_name))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 1. Thực thi Gaussian Blur
        t1 = time.time()
        res_gaussian = apply_gaussian_blur(img, S_S)
        dt_g = time.time() - t1

        # 2. Thực thi Standard Bilateral Filter (Làm Ground Truth chuẩn đoán sai số)
        t2 = time.time()
        res_standard = apply_standard_bilateral(img, S_S, S_R)
        dt_s = time.time() - t2

        # 3. Thực thi Constant Time O(1) Phiên bản gốc (Base)
        t3 = time.time()
        res_base_rgb = ct_base.apply(img_rgb)
        res_base = cv2.cvtColor(res_base_rgb, cv2.COLOR_RGB2BGR)
        dt_o1_base = time.time() - t3

        # 4. Thực thi Constant Time O(1) Phiên bản tối ưu nghiêm ngặt (Strict)
        t4 = time.time()
        res_strict_rgb = ct_strict.apply(img_rgb)
        res_strict = cv2.cvtColor(res_strict_rgb, cv2.COLOR_RGB2BGR)
        dt_o1_strict = time.time() - t4

        # 5. Tính toán độ chính xác đồ họa đối chiếu với Ground Truth (Standard)
        psnr_base = psnr(res_standard, res_base)
        ssim_base = ssim(res_standard, res_base, channel_axis=2)

        psnr_strict = psnr(res_standard, res_strict)
        ssim_strict = ssim(res_standard, res_strict, channel_axis=2)

        # In dữ liệu so sánh trực quan ra màn hình console
        print(f"{f_name[:18]:<18} | {dt_g:<9.4f}s | {dt_s:<9.4f}s | "
              f"{dt_o1_base:<14.4f}s | {dt_o1_strict:<16.4f}s | "
              f"{psnr_base:<9.2f}dB | {psnr_strict:<11.2f}dB | {ssim_strict:<11.4f}")

        # --- GHI DỮ LIỆU ĐỐI CHIẾU CHI TIẾT VÀO FILE CSV ---
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(csv_file_path, mode='a', newline='', encoding='utf-8') as csv_f:
            writer = csv.writer(csv_f)
            writer.writerow([
                current_time, f_name, S_S, S_R,
                f"{dt_g:.6f}", f"{dt_s:.6f}", f"{dt_o1_base:.6f}", f"{dt_o1_strict:.6f}",
                f"{psnr_base:.4f}", f"{psnr_strict:.4f}", f"{ssim_base:.4f}", f"{ssim_strict:.4f}"
            ])
        # ------------------------------------------------------------

        # --- TẠO ẢNH MA TRẬN SO SÁNH 5 KHUNG NGANG VÀ HEADER ---
        h, w = img.shape[:2]
        display_w = 350  # Thu nhỏ kích thước hiển thị thành phần để vừa màn hình khi ghép 5 ảnh
        display_h = int(h * (display_w / w))

        def resize_img(im):
            return cv2.resize(im, (display_w, display_h))

        # Ghép đồng thời 5 trường ảnh để kiểm tra trực quan sai khác biên cạnh
        content_row = cv2.hconcat([
            resize_img(img),
            resize_img(res_gaussian),
            resize_img(res_standard),
            resize_img(res_base),
            resize_img(res_strict)
        ])

        header_h = 80
        header = np.zeros((header_h, content_row.shape[1], 3), dtype=np.uint8)

        names = [
            "Original",
            "Gaussian Blur",
            "Standard Bilateral",
            "O(1) Base",
            "O(1) Strict"
        ]

        times = [
            "",
            f"{dt_g:.4f}s",
            f"{dt_s:.4f}s",
            f"{dt_o1_base:.4f}s",
            f"{dt_o1_strict:.4f}s"
        ]

        for i in range(5):
            x_pos = i * display_w + 10

            # Tên thuật toán
            cv2.putText(header, names[i], (x_pos, 28), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            # Thời gian chạy tương ứng
            if times[i]:
                cv2.putText(header, times[i], (x_pos, 60), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

        final_comparison = cv2.vconcat([header, content_row])
        save_path = os.path.join(output_dir, f"compare_{f_name}")
        cv2.imwrite(save_path, final_comparison)

    print("=" * 160)
    print(f"\n[XONG] Toàn bộ dữ liệu đo kiểm cấu trúc và thời gian đã kết xuất tại: {csv_file_path}")


if __name__ == "__main__":
    main()