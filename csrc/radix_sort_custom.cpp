#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cstdint>

namespace py = pybind11;

// -----------------------------
struct Pixel {
    uint8_t r, g, b;
};

static inline uint8_t get_dim(const Pixel& p, int dim) {
    return (dim == 0) ? p.r : (dim == 1 ? p.g : p.b);
}

// -----------------------------
// counting sort by 1 dimension
// -----------------------------
std::vector<Pixel> sort_by_dim(const std::vector<Pixel>& arr, int dim) {
    const int K = 256;

    std::vector<int> cnt(K, 0);

    for (const auto& p : arr) {
        cnt[get_dim(p, dim)]++;
    }

    std::vector<int> pos(K, 0);
    for (int i = 1; i < K; i++) {
        pos[i] = pos[i - 1] + cnt[i - 1];
    }

    std::vector<Pixel> out(arr.size());

    for (const auto& p : arr) {
        int k = get_dim(p, dim);
        out[pos[k]++] = p;
    }

    return out;
}

// -----------------------------
// Python API
// -----------------------------
py::array_t<uint8_t> sort_by_dim_py(py::array_t<uint8_t> input, int dim) {
    auto buf = input.unchecked<2>();

    std::vector<Pixel> arr;
    arr.reserve(buf.shape(0));

    for (ssize_t i = 0; i < buf.shape(0); i++) {
        arr.push_back({
            buf(i, 0),
            buf(i, 1),
            buf(i, 2)
        });
    }

    auto sorted = sort_by_dim(arr, dim);

    py::array_t<uint8_t> out({(int)sorted.size(), 3});
    auto r = out.mutable_unchecked<2>();

    for (size_t i = 0; i < sorted.size(); i++) {
        r(i, 0) = sorted[i].r;
        r(i, 1) = sorted[i].g;
        r(i, 2) = sorted[i].b;
    }

    return out;
}

// -----------------------------
PYBIND11_MODULE(radix_sort_custom, m) {
    m.def("sort_by_dim", &sort_by_dim_py,
          "Radix/counting sort RGB by dimension (0/1/2)");
}