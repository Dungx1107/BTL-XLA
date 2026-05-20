#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>

namespace py = pybind11;

static inline float dist2(
    const uint8_t* a,
    const uint8_t* b
) {
    float dr = float(a[0]) - float(b[0]);
    float dg = float(a[1]) - float(b[1]);
    float db = float(a[2]) - float(b[2]);
    return dr*dr + dg*dg + db*db;
}

py::array_t<uint8_t> merge_colors(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> input,
    float threshold
) {
    auto buf = input.request();
    auto* ptr = static_cast<uint8_t*>(buf.ptr);

    int N = buf.shape[0];
    float t2 = threshold * threshold;

    std::vector<bool> used(N, false);

    std::vector<std::array<float,3>> out;
    out.reserve(N);

    for (int i = 0; i < N; i++) {
        if (used[i]) continue;

        float acc[3] = {
            float(ptr[i * 3 + 0]),
            float(ptr[i * 3 + 1]),
            float(ptr[i * 3 + 2])
        };

        int count = 1;
        used[i] = true;

        for (int j = i + 1; j < N; j++) {
            if (used[j]) continue;

            if (dist2(&ptr[i * 3], &ptr[j * 3]) < t2) {
                acc[0] += ptr[j * 3 + 0];
                acc[1] += ptr[j * 3 + 1];
                acc[2] += ptr[j * 3 + 2];

                used[j] = true;
                count++;
            }
        }

        acc[0] /= count;
        acc[1] /= count;
        acc[2] /= count;

        out.push_back({acc[0], acc[1], acc[2]});
    }

    py::array_t<uint8_t> result({(int)out.size(), 3});
    auto r = result.mutable_unchecked<2>();

    for (size_t i = 0; i < out.size(); i++) {
        r(i, 0) = (uint8_t)out[i][0];
        r(i, 1) = (uint8_t)out[i][1];
        r(i, 2) = (uint8_t)out[i][2];
    }

    return result;
}

PYBIND11_MODULE(color_merge_cpp, m) {
    m.def("merge", &merge_colors, "O(K^2) RGB merge (uint8 numpy)");
}