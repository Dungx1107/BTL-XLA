from setuptools import setup, Extension
import pybind11
import sysconfig

ext_modules = [
    Extension(
        "radix_sort_custom",
        ["csrc/radix_sort_custom.cpp"],
        include_dirs=[
            pybind11.get_include(),
            sysconfig.get_path("include"),
        ],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
    ),

    Extension(
        "color_merge_cpp",
        ["csrc/color_merge.cpp"],
        include_dirs=[
            pybind11.get_include(),
            sysconfig.get_path("include"),
        ],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
    ),
]

setup(
    name="custom_cpp_ops",
    version="0.0.1",
    ext_modules=ext_modules,
)