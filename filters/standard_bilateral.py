import cv2

def apply_standard_bilateral(image, sigma_s, sigma_r):
    # d = đường kính vùng lân cận
    d = int(sigma_s * 2)
    return cv2.bilateralFilter(image, d, sigmaColor=sigma_r*255, sigmaSpace=sigma_s)