import cv2
import time
from core import O1BilateralFilter


def main():
    # Đọc ảnh đầu vào từ thư mục images
    # Bạn cần copy file ảnh vào thư mục images/ trước khi chạy
    img = cv2.imread('images/hibiscus.jpg')
    if img is None:
        print("Vui lòng bỏ file 'hibiscus.jpg' vào thư mục 'images'")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Khởi tạo bộ lọc với N=20 màu mẫu [cite: 248]
    bf = O1BilateralFilter(n_samples=20, sigma_s=15, sigma_r=0.1)

    start = time.time()
    result = bf.apply(img_rgb)
    print(f"Thời gian xử lý: {time.time() - start:.4f} giây")

    # Hiển thị và lưu kết quả
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite('results/output.png', result_bgr)
    cv2.imshow('Original', img)
    cv2.imshow('O(1) Bilateral Filter', result_bgr)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()