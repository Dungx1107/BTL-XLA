#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

double calculatePSNR(const string& img1_path, const string& img2_path) {
    // Đọc ảnh
    Mat img1 = imread(img1_path);
    Mat img2 = imread(img2_path);

    if (img1.empty() || img2.empty()) {
        cerr << "Lỗi: Không thể tải ảnh." << endl;
        return -1.0;
    }

    if (img1.rows != img2.rows || img1.cols != img2.cols || img1.type() != img2.type()) {
        cerr << "Lỗi: Kích thước hoặc định dạng của hai ảnh không khớp." << endl;
        return -1.0;
    }

    Mat s1;
    // Lấy giá trị tuyệt đối của hiệu hai ma trận: |img1 - img2|
    absdiff(img1, img2, s1);

    // Ép kiểu sang Float 32-bit để bình phương không bị tràn dải 8-bit
    s1.convertTo(s1, CV_32F);

    // Bình phương từng phần tử: |img1 - img2|^2
    s1 = s1.mul(s1);

    // Tính tổng các sai số bình phương theo từng kênh màu
    Scalar s = sum(s1);
    double sse = s.val[0] + s.val[1] + s.val[2];

    // Nếu Sum of Squared Errors tiến tới 0 (ảnh giống nhau hoàn toàn)
    if (sse <= 1e-10) {
        return INFINITY;
    } else {
        // Tính MSE = Tổng sai số / (Tổng số pixel * Số kênh màu)
        double mse = sse / (double)(img1.channels() * img1.total());

        // Tính PSNR với MAX_I = 255
        double psnr = 10.0 * log10((255 * 255) / mse);
        return psnr;
    }
}

int main() {
    /* string path_original = "original.png";
    string path_processed = "processed.png";

    double psnr_val = calculatePSNR(path_original, path_processed);
    if (psnr_val != -1.0) {
        cout << "PSNR: " << psnr_val << " dB" << endl;
    }
    */
    return 0;
}