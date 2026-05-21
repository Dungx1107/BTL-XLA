import cv2
import numpy as np
import math


def calculate_psnr(img1_path: str, img2_path: str) -> float:
    # Đọc ảnh (định dạng mặc định BGR của OpenCV)
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        raise FileNotFoundError("Không thể đọc được ảnh từ đường dẫn cung cấp.")

    # Kiểm tra kích thước ma trận ảnh
    if img1.shape != img2.shape:
        raise ValueError("Hai ảnh phải có cùng kích thước (width, height, channels).")

    # Ép kiểu dữ liệu sang float64 để tránh tràn bộ nhớ (overflow) khi tính bình phương
    img1_float = img1.astype(np.float64)
    img2_float = img2.astype(np.float64)

    # Tính Mean Squared Error (MSE) cho toàn bộ ma trận
    mse = np.mean((img1_float - img2_float) ** 2)

    # Nếu MSE = 0, hai ảnh giống hệt nhau, PSNR tiến tới vô cực
    if mse == 0:
        return float('inf')

    # Giá trị pixel tối đa của ảnh 8-bit
    max_pixel = 255.0

    # Áp dụng công thức tính PSNR
    psnr = 10 * math.log10((max_pixel ** 2) / mse)
    return psnr

# Cách gọi hàm
# psnr_value = calculate_psnr("original.png", "processed.png")
# print(f"PSNR: {psnr_value:.2f} dB")