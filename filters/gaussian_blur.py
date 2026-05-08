import cv2

def apply_gaussian_blur(image, sigma_s):
    # Kích thước kernel tính dựa trên sigma_s
    k_size = int(sigma_s * 3) | 1 # Đảm bảo là số lẻ
    return cv2.GaussianBlur(image, (k_size, k_size), sigma_s)