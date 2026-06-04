import cv2


def apply_standard_bilateral(image, sigma_s, sigma_r):
    """
    Áp dụng bộ lọc Bilateral tiêu chuẩn làm Ground Truth để tính toán PSNR.
    - image: Ảnh đầu vào (uint8, BGR hoặc RGB)
    - sigma_s: Độ lệch chuẩn không gian (Spatial sigma)
    - sigma_r: Độ lệch chuẩn dải màu (Range sigma, dải từ 0.0 đến 1.0)
    """
    sigma_color_uint8 = float(sigma_r * 255.0)
    d = -1
    return cv2.bilateralFilter(
        src=image,
        d=d,
        sigmaColor=sigma_color_uint8,
        sigmaSpace=float(sigma_s)
    )