from benchmark import FilterBenchmark

from filters.constant_time_bilateral_2 import ConstantTimeBilateral as CTB_Version2
from filters.constant_time_bilateral_5 import VarianceSplitVer


def main():
    # 1. Khởi tạo bộ công cụ benchmark
    tester = FilterBenchmark(input_dir='images', output_dir='results3')

    # Tham số cấu hình bộ lọc
    S_S = 35
    S_R = 0.08

    # Khởi tạo instance cho các class lọc của bạn
    ct_paper_v2 = CTB_Version2(n_samples=20, sigma_s=S_S, sigma_r=S_R)
    ct_variance_split = VarianceSplitVer(sigma_s=S_S, sigma_r=S_R)

    # 2. Đăng ký các phương pháp lọc vào hệ thống.
    # so sánh phương pháp gốc và phương pháp cải tiến
    tester.register_filter('Original filter', lambda img: ct_paper_v2.apply(img))
    tester.register_filter('New filter', lambda img: ct_variance_split.apply(img))

    # 3. Tiến hành chạy benchmark toán diện
    tester.run(ground_truth_filter_name='Standard Bilateral')


if __name__ == '__main__':
    main()