# Tài liệu dự án Bilateral_O1

## 1. Tổng quan

Dự án này tập trung vào nghiên cứu, triển khai và đánh giá các phương pháp lọc ảnh Bilateral. Mục tiêu chính là so sánh:
- lọc Gaussian đơn giản,
- lọc Bilateral tiêu chuẩn của OpenCV,
- các biến thể "constant-time" (O(1)) theo ý tưởng của bài báo,
- và các cải tiến lấy mẫu màu, chia vùng và ghép màu.

Dự án gồm nhiều phiên bản thuật toán trong thư mục `filters/`, cùng các công cụ benchmark, so sánh hình ảnh và tiện ích tính PSNR để đánh giá chất lượng.

## 2. Hướng dẫn sử dụng nhanh

### 2.1 Cài đặt môi trường

1. Tạo và kích hoạt môi trường Python.
2. Cài đặt dependency:
   - `pip install -r requirements.txt`
3. Build extension C++ nếu cần:
   - `scripts/build.sh`
   - hoặc `pip install -e . --no-build-isolation`

### 2.2 Chạy các bài toán chính

- `python main.py`
  - Chạy so sánh trực quan giữa ảnh gốc, Gaussian blur, Bilateral tiêu chuẩn và biến thể O(1).
- `python run_benchmark.py`
  - Chạy benchmark giữa biến thể gốc và biến thể mới, lưu kết quả vào thư mục `results3`.
- `python test_output.py`
  - Tạo ảnh so sánh 3 cột cho ảnh gốc, bộ lọc gốc và bộ lọc mới.
- `python test_v2.py`
  - Thực nghiệm lấy mẫu palette và trực quan hóa gói màu từ các vùng ảnh.
- `benchmark_bilateral.ipynb`
  - Notebook tương tác để phân tích và kiểm thử trực quan.

## 3. Cấu trúc thư mục chính

- `main.py`
- `benchmark.py`
- `run_benchmark.py`
- `test.py`
- `test_v2.py`
- `test_output.py`
- `setup.py`
- `requirements.txt`
- `benchmark_bilateral.ipynb`
- `filters/`
- `csrc/`
- `utils/`
- `scripts/`
- `images/` - ảnh đầu vào để thử nghiệm.
- `results/`, `results2/` - thư mục kết quả và benchmark.

## 4. Chi tiết từng file và thư mục

### 4.1 `main.py`

- Là kịch bản chính để chạy thử nghiệm so sánh.
- Đọc ảnh từ thư mục `images/`.
- Áp dụng:
  - Gaussian blur (`filters/gaussian_blur.py`),
  - Bilateral tiêu chuẩn (`filters/standard_bilateral.py`),
  - Biến thể O(1) ban đầu (`filters/constant_time_bilateral.py`).
- Lưu ảnh so sánh ghép 4 cột vào `results/`.
- In thời gian chạy của từng phương pháp ra console.

### 4.2 `benchmark.py`

- Chứa lớp `FilterBenchmark` để benchmark chung.
- Chức năng chính:
  - đăng ký các phương pháp lọc bằng `register_filter(name, filter_func)`;
  - duyệt ảnh trong `input_dir`;
  - đo thời gian chạy và lưu ảnh đầu ra;
  - tính PSNR so sánh với ảnh gốc hoặc ảnh chuẩn nếu `ground_truth_filter_name` được chỉ định.
- Dùng như một thành phần tái sử dụng cho các script benchmark.

### 4.3 `run_benchmark.py`

- Ví dụ sử dụng `FilterBenchmark` để so sánh hai bộ lọc:
  - `ConstantTimeBilateral` từ `filters/constant_time_bilateral_2.py`,
  - `VarianceSplitVer` từ `filters/constant_time_bilateral_5.py`.
- Kết quả được lưu vào thư mục `results3`.
- Lưu ý: script hiện gọi `ground_truth_filter_name='Standard Bilateral'` nhưng file không đăng ký phương pháp này, nên PSNR được so sánh với ảnh gốc theo cấu hình hiện tại.

### 4.4 `test.py`

- Chứa hàm thử nghiệm lấy mẫu màu và trực quan hóa palette.
- Bao gồm:
  - `origin_color_sampling()` - lấy mẫu màu theo phương pháp Poisson Disk gốc,
  - `_split()` và `select_colors()` - chia màu theo không gian RGB với radix sort và gom màu,
  - `draw_palette()` - dựng lưới palette cho trực quan,
  - `run()` - xuất ảnh so sánh palette giữa phương pháp gốc và phương pháp mới.
- Thích hợp để kiểm tra chiến lược chọn tập mẫu màu.

### 4.5 `test_v2.py`

- Thử nghiệm cách chia palette bằng octree/variance split.
- Chứa các hàm:
  - `_variance()` - tính phương sai của tập điểm màu,
  - `_split()` - chia dải màu thành nhiều phần theo độ biến đổi,
  - `get_palette()` - tạo palette cho một patch ảnh,
  - `build_debug_image()` - tạo ảnh debug gồm patch và palette tương ứng.
- Dùng để phân tích tính hiệu quả của palette theo vùng ảnh.

### 4.6 `test_output.py`

- Tạo ảnh so sánh trực quan 3 cột:
  - ảnh gốc,
  - kết quả bộ lọc gốc (`filters/constant_time_bilateral_2.py`),
  - kết quả bộ lọc mới (`filters/constant_time_bilateral_4.py`).
- Dùng để so sánh trực quan lời giải và biến thể nâng cao.

### 4.7 `setup.py`

- Định nghĩa hai tiện ích mở rộng C++:
  - `radix_sort_custom` từ `csrc/radix_sort_custom.cpp`,
  - `color_merge_cpp` từ `csrc/color_merge.cpp`.
- Sử dụng `pybind11` để kết nối code C++ với Python.
- Thường dùng khi cài đặt package ở chế độ editable.

### 4.8 `requirements.txt`

- Danh sách thư viện cần thiết:
  - `pybind11`,
  - `numpy`,
  - `Pillow`,
  - `setuptools`.

### 4.9 `filters/` - các bộ lọc chính

#### `filters/gaussian_blur.py`

- Áp dụng Gaussian Blur OpenCV.
- Dùng làm đối chiếu tốc độ và chất lượng đơn giản.
- Hàm chính: `apply_gaussian_blur(image, sigma_s)`.

#### `filters/standard_bilateral.py`

- Áp dụng Bilateral Filter tiêu chuẩn của OpenCV.
- Hàm chính: `apply_standard_bilateral(image, sigma_s, sigma_r)`.
- Dùng làm ground truth so sánh chất lượng.

#### `filters/constant_time_bilateral.py`

- Phiên bản O(1) ban đầu.
- Các bước:
  1. Lấy mẫu màu Poisson Disk từ ảnh,
  2. Tính các stack lọc không gian bằng `cv2.boxFilter`,
  3. Nội suy kết quả bằng KD-Tree giữa màu điểm ảnh và các mẫu.
- Là phiên bản mẫu để so sánh và hiểu luồng thuật toán.

#### `filters/constant_time_bilateral_2.py`

- Phiên bản tái cấu trúc của `constant_time_bilateral.py`.
- Tách rõ 3 giai đoạn thành phương thức riêng:
  - `_poisson_disk_sampling()`,
  - `_spatial_filter_stacks()`,
  - `_interpolate_results()`.
- Giữ nguyên logic O(1) nhưng dễ đọc hơn.

#### `filters/constant_time_bilateral_3.py`

- Phiên bản "strict" với chú thích thuật toán chặt chẽ.
- Thực hiện lọc và nội suy theo cách vectorized hơn.
- Mục tiêu hướng tới minh hoạ nghiêm ngặt hơn về các công thức toán học trong bài báo.

#### `filters/constant_time_bilateral_4.py`

- `BoxSplitVer` kế thừa `ConstantTimeBilateral` từ phiên bản 2.
- Sử dụng `radix_sort_custom` để phân chia màu theo dim RGB,
- Dùng `color_merge_cpp` để gom palette và giảm số màu.
- Là một bước thử nghiệm cải tiến cho bước lấy mẫu màu.

#### `filters/constant_time_bilateral_5.py`

- `VarianceSplitVer` là phiên bản nâng cao nhất trong dự án.
- Sử dụng cách chia patch theo tile (`tile_size`) và xử lý từng block nhỏ.
- Thực hiện lấy mẫu màu bằng phân chia octree dựa trên phương sai và gom màu bằng `color_merge_cpp`.
- Dùng lại logic lọc O(1) từ `filters/constant_time_bilateral_2.py` để xử lý mỗi tile.
- Phù hợp cho ảnh lớn và tối ưu hóa bằng chia vùng.

### 4.10 `csrc/` - mã C++ hỗ trợ

#### `csrc/radix_sort_custom.cpp`

- Cài đặt hàm `sort_by_dim` để sắp xếp pixel theo một nhánh màu R/G/B.
- Dùng trong `filters/constant_time_bilateral_4.py` và thử nghiệm chia màu.
- Tăng tốc việc phân chia tập màu bằng C++.

#### `csrc/color_merge.cpp`

- Cài đặt hàm `merge` để gom các màu gần nhau thành một màu trung bình.
- Dùng trong các bước xây palette để giảm số mẫu màu và tăng tính ổn định.

### 4.11 `utils/psnr.py`

- Cung cấp hàm `calculate_psnr(img1_path, img2_path)` để tính giá trị PSNR giữa hai ảnh.
- Dùng cho đánh giá chất lượng ảnh đầu ra.

### 4.12 `scripts/build.sh`

- Script đơn giản chạy:
  - `pip install -e . --no-build-isolation`
- Dùng để cài đặt package và build các extension C++.

## 5. Ghi chú quan trọng

- Thư mục `images/` chứa ảnh đầu vào mẫu để thử nghiệm.
- `results/`, `results2/`, `results3/` là thư mục lưu ảnh kết quả.
- `benchmark.py` là thành phần chung, không phải script chạy độc lập.
- Một số script hiện tại là demo/so sánh nội bộ, không phải pipeline hoàn chỉnh.
- Nếu muốn so sánh PSNR với bộ lọc Bilateral tiêu chuẩn, cần đăng ký filter đó trong `run_benchmark.py` với cùng tên `Standard Bilateral`.

---

## 6. Đề xuất mở rộng

- Thêm file README ở root để hướng dẫn nhanh hơn.
- Chuẩn hóa tên filter khi benchmark để sử dụng `ground_truth_filter_name` không lỗi.
- Bổ sung các ví dụ chạy với tham số `sigma_s`, `sigma_r` khác nhau.
- Ghi chú thêm về cách build C++ extension trên Windows nếu cần.
