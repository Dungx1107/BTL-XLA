import cv2
from benchmark import FilterBenchmark

# Import các thuật toán từ kiến trúc hiện tại của bạn
from filters.gaussian_blur import apply_gaussian_blur
from filters.standard_bilateral import apply_standard_bilateral
from filters.constant_time_bilateral import ConstantTimeBilateral
# Khai báo thêm các phiên bản thử nghiệm khác của bạn
from filters.constant_time_bilateral_2 import ConstantTimeBilateral as CTB_Version2
from filters.constant_time_bilateral_5 import VarianceSplitVer


def main():
    # 1. Khởi tạo bộ công cụ benchmark
    tester = FilterBenchmark(input_dir='images', output_dir='results3')

    # Tham số cấu hình bộ lọc
    S_S = 35
    S_R = 0.08

    # Khởi tạo instance cho các class lọc của bạn
    ct_paper_v1 = ConstantTimeBilateral(n_samples=20, sigma_s=S_S, sigma_r=S_R)
    ct_paper_v2 = CTB_Version2(n_samples=20, sigma_s=S_S, sigma_r=S_R)
    ct_variance_split = VarianceSplitVer(sigma_s=S_S, sigma_r=S_R)

    # 2. Đăng ký các phương pháp lọc vào hệ thống.
    # Sử dụng hàm lambda để wrap các hàm/phương thức có cấu trúc gọi khác nhau về một chuẩn nhận ảnh chung: lambda img: ...
    tester.register_filter('Gaussian Blur', lambda img: apply_gaussian_blur(img))

    tester.register_filter('Standard Bilateral', lambda img: apply_standard_bilateral(img))

    tester.register_filter('O(1) Paper V1', lambda img: ct_paper_v1.apply(img))

    tester.register_filter('O(1) Paper V2', lambda img: ct_paper_v2.apply(img))

    tester.register_filter('O(1) Var Split', lambda img: ct_variance_split.apply(img))

    # 3. Tiến hành chạy benchmark toán diện
    # Bạn có thể đặt 'Standard Bilateral' làm ground_truth_filter_name để đo sai số toán học
    # của các thuật toán xấp xỉ O(1) so với công thức chuẩn của bộ lọc lõi.
    tester.run(ground_truth_filter_name='Standard Bilateral')


if __name__ == '__main__':
    main()