import cv2
import numpy as np
import time
import os
from filters.constant_time_bilateral import ConstantTimeBilateral
from filters.standard_bilateral import apply_standard_bilateral
from filters.gaussian_blur import apply_gaussian_blur


def main():
    input_dir = 'images'
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)

    # Cấu hình tham số
    S_S = 35
    S_R = 0.08

    img_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    ct_bilateral = ConstantTimeBilateral(n_samples=20, sigma_s=S_S, sigma_r=S_R)

    # Header cho Console
    print("\n" + "=" * 85)
    print(f"{'IMAGE NAME':<25} | {'GAUSSIAN':<15} | {'STANDARD':<15} | {'O(1) PAPER':<15}")
    print("-" * 85)

    for f in img_files:
        img = cv2.imread(os.path.join(input_dir, f))
        if img is None: continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Thực thi các thuật toán
        t1 = time.time()
        res_gaussian = apply_gaussian_blur(img, S_S)
        dt_g = time.time() - t1

        t2 = time.time()
        res_standard = apply_standard_bilateral(img, S_S, S_R)
        dt_s = time.time() - t2

        t3 = time.time()
        res_o1_rgb = ct_bilateral.apply(img_rgb)
        res_o1 = cv2.cvtColor(res_o1_rgb, cv2.COLOR_RGB2BGR)
        dt_o1 = time.time() - t3

        # In kết quả ra console theo format đẹp
        print(f"{f[:25]:<25} | {dt_g:<14.4f}s | {dt_s:<14.4f}s | {dt_o1:<14.4f}s")

        # --- TẠO ẢNH SO SÁNH CÓ HEADER TÁCH BIỆT ---
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

        # Tạo một dải băng đen phía trên (chiều cao 50px)
        header_h = 50
        header = np.zeros((header_h, content_row.shape[1], 3), dtype=np.uint8)

        # Thêm text vào dải băng đen
        labels = ["Original", "Gaussian (Blur)", "Standard (Slow)", "O(1) Paper"]
        for i, label in enumerate(labels):
            x_pos = i * display_w + 10
            cv2.putText(header, label, (x_pos, 35), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        # Ghép header vào nội dung
        final_comparison = cv2.vconcat([header, content_row])

        # Lưu kết quả
        save_path = os.path.join(output_dir, f"compare_{f}")
        cv2.imwrite(save_path, final_comparison)

    print("=" * 85)
    print(f"\n[XONG] Tất cả kết quả so sánh đã được lưu tại: {output_dir}")


if __name__ == "__main__":
    main()