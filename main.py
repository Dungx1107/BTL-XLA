import cv2
import numpy as np
import time
import os
import csv
from datetime import datetime
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from filters.constant_time_bilateral import ConstantTimeBilateral
from filters.standard_bilateral import apply_standard_bilateral
from filters.gaussian_blur import apply_gaussian_blur


def main():
    input_dir = 'images'
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)

    # Cấu hình tham số bộ lọc
    S_S = 35
    S_R = 0.08

    img_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    ct_bilateral = ConstantTimeBilateral(n_samples=20, filter_radius=S_S, sigma_r=S_R)

    # --- CẤU HÌNH FILE CSV BỔ SUNG CÁC CỘT ĐỘ CHÍNH XÁC ---
    csv_file_path = os.path.join(output_dir, 'csv/complete_benchmark.csv')
    csv_headers = [
        "Timestamp", "Image_Name", "Sigma_S", "Sigma_R",
        "Gaussian_Time (s)", "Standard_Time (s)", "O1_Paper_Time (s)",
        "O1_vs_Standard_PSNR (dB)", "O1_vs_Standard_SSIM"
    ]

    # Tạo thư mục csv nếu chưa tồn tại
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
    # -----------------------------------------------------

    # Header cho Console
    print("\n" + "=" * 120)
    print(
        f"{'IMAGE NAME':<20} | {'GAUSSIAN':<12} | {'STANDARD':<12} | {'O(1) PAPER':<12} | {'PSNR (O1)':<12} | {'SSIM (O1)':<12}")
    print("-" * 120)

    for f_name in img_files:
        img = cv2.imread(os.path.join(input_dir, f_name))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 1. Thực thi thuật toán Gaussian Blur và đo thời gian
        t1 = time.time()
        res_gaussian = apply_gaussian_blur(img, S_S)
        dt_g = time.time() - t1

        # 2. Thực thi thuật toán Standard Bilateral Filter và đo thời gian (Làm Ground Truth)
        t2 = time.time()
        res_standard = apply_standard_bilateral(img, S_S, S_R)
        dt_s = time.time() - t2

        # 3. Thực thi thuật toán Constant Time O(1) Bilateral Filter và đo thời gian
        t3 = time.time()
        res_o1_rgb = ct_bilateral.apply(img_rgb)
        res_o1 = cv2.cvtColor(res_o1_rgb, cv2.COLOR_RGB2BGR)
        dt_o1 = time.time() - t3

        # 4. Tính toán độ chính xác đồ họa (PSNR & SSIM) trực tiếp trên ma trận gốc chưa nén/resize
        val_psnr = psnr(res_standard, res_o1)
        val_ssim = ssim(res_standard, res_o1, channel_axis=2)

        # In kết quả ra console theo format mở rộng
        print(
            f"{f_name[:20]:<20} | {dt_g:<11.4f}s | {dt_s:<11.4f}s | {dt_o1:<11.4f}s | {val_psnr:<10.2f}dB | {val_ssim:<10.4f}")

        # --- GHI ĐẦY ĐỦ THỜI GIAN CHẠY VÀ ĐỘ ĐO CHÍNH XÁC VÀO CSV ---
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(csv_file_path, mode='a', newline='', encoding='utf-8') as csv_f:
            writer = csv.writer(csv_f)
            writer.writerow([
                current_time, f_name, S_S, S_R,
                f"{dt_g:.6f}", f"{dt_s:.6f}", f"{dt_o1:.6f}",
                f"{val_psnr:.4f}", f"{val_ssim:.4f}"
            ])
        # ------------------------------------------------------------

        # --- TẠO ẢNH SO SÁNH CÓ HEADER TÁCH BIỆT (GIỮ NGUYÊN HOÀN TOÀN NHƯ CŨ) ---
        h, w = img.shape[:2]
        display_w = 400
        display_h = int(h * (display_w / w))

        def resize_img(im):
            return cv2.resize(im, (display_w, display_h))

        # Ghép 4 ảnh ngang
        content_row = cv2.hconcat([
            resize_img(img),
            resize_img(res_gaussian),
            resize_img(res_standard),
            resize_img(res_o1)
        ])

        # Tạo một dải băng đen phía trên
        header_h = 80
        header = np.zeros((header_h, content_row.shape[1], 3), dtype=np.uint8)

        names = [
            "Original",
            "Gaussian Blur",
            "Standard Bilateral",
            "O(1) Paper"
        ]

        times = [
            "",
            f"{dt_g:.4f}s",
            f"{dt_s:.4f}s",
            f"{dt_o1:.4f}s"
        ]

        for i in range(4):
            x_pos = i * display_w + 10

            # Tên thuật toán
            cv2.putText(
                header,
                names[i],
                (x_pos, 28),
                cv2.FONT_HERSHEY_DUPLEX,
                0.65,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            # Thời gian chạy
            if times[i]:
                cv2.putText(
                    header,
                    times[i],
                    (x_pos, 60),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.65,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA
                )

        # Ghép header vào nội dung
        final_comparison = cv2.vconcat([header, content_row])

        # Lưu kết quả
        save_path = os.path.join(output_dir, f"compare_{f_name}")
        cv2.imwrite(save_path, final_comparison)

    print("=" * 120)
    print(f"\n[XONG] Tất cả kết quả thực thi và sai số đã được ghi nhận tại: {csv_file_path}")


if __name__ == "__main__":
    main()