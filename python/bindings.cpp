#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "pixmask/api.h"
#include "pixmask/image.h"
#include "pixmask/security.h"

namespace py = pybind11;

namespace {

enum class OutputType {
    Auto,
    Uint8,
    Float32,
};

struct ArrayShape {
    std::size_t height = 0;
    std::size_t width = 0;
    std::size_t channels = 0;
};

ArrayShape validate_shape(const py::buffer_info &info) {
    if (info.ndim != 3) {
        throw py::value_error("expected an array with shape (H, W, C)");
    }

    ArrayShape shape{};
    shape.height = static_cast<std::size_t>(info.shape[0]);
    shape.width = static_cast<std::size_t>(info.shape[1]);
    shape.channels = static_cast<std::size_t>(info.shape[2]);

    if (shape.height == 0 || shape.width == 0) {
        throw py::value_error("image dimensions must be positive");
    }

    if (shape.channels != 3 && shape.channels != 4) {
        throw py::value_error("channel dimension must be 3 (RGB) or 4 (RGBA)");
    }

    if ((shape.width % 2u) != 0u || (shape.height % 2u) != 0u) {
        throw py::value_error("image width and height must be even");
    }

    return shape;
}

template <typename T>
bool is_c_contiguous(const py::buffer_info &info, std::size_t channels) noexcept {
    return static_cast<std::size_t>(info.strides[2]) == sizeof(T) &&
           static_cast<std::size_t>(info.strides[1]) == channels * sizeof(T) &&
           static_cast<std::size_t>(info.strides[0]) ==
               static_cast<std::size_t>(info.shape[1]) * channels * sizeof(T);
}

template <typename T>
std::vector<T> copy_buffer(const py::buffer_info &info, const ArrayShape &shape, std::size_t channels) {
    if (!is_c_contiguous<T>(info, channels)) {
        throw py::value_error("input array must be C-contiguous");
    }

    const auto *src = static_cast<const T *>(info.ptr);
    const std::size_t total = shape.height * shape.width * channels;
    std::vector<T> buffer(total);
    if (total > 0) {
        std::memcpy(buffer.data(), src, total * sizeof(T));
    }
    return buffer;
}

py::array create_output_array(py::dtype dtype, const ArrayShape &shape) {
    return py::array(dtype, {static_cast<py::ssize_t>(shape.height),
                              static_cast<py::ssize_t>(shape.width),
                              static_cast<py::ssize_t>(3)});
}

std::vector<std::uint8_t> drop_alpha(const std::vector<std::uint8_t> &rgba, std::size_t pixels) {
    std::vector<std::uint8_t> rgb(pixels * 3u);
    for (std::size_t i = 0; i < pixels; ++i) {
        const std::size_t src_base = i * 4u;
        const std::size_t dst_base = i * 3u;
        rgb[dst_base + 0] = rgba[src_base + 0];
        rgb[dst_base + 1] = rgba[src_base + 1];
        rgb[dst_base + 2] = rgba[src_base + 2];
    }
    return rgb;
}

py::array sanitize_array(py::array input, py::kwargs kwargs) {
    OutputType output_pref = OutputType::Auto;
    for (auto &item : kwargs) {
        const std::string key = py::cast<std::string>(item.first);
        if (key == "output_dtype") {
            const std::string value = py::cast<std::string>(item.second);
            if (value == "uint8") {
                output_pref = OutputType::Uint8;
            } else if (value == "float32") {
                output_pref = OutputType::Float32;
            } else {
                throw py::value_error("output_dtype must be 'uint8' or 'float32'");
            }
        } else {
            throw py::type_error("unexpected keyword argument '" + key + "'");
        }
    }

    const py::buffer_info info = input.request();
    const ArrayShape shape = validate_shape(info);

    const bool is_uint8 = info.format == py::format_descriptor<std::uint8_t>::format();
    const bool is_float32 = info.format == py::format_descriptor<float>::format();

    pixmask::PixelType input_type = pixmask::PixelType::U8_RGB;
    std::vector<std::uint8_t> storage_u8;
    std::vector<float> storage_f32;

    if (is_uint8) {
        storage_u8 = copy_buffer<std::uint8_t>(info, shape, shape.channels);
        if (shape.channels == 4) {
            storage_u8 = drop_alpha(storage_u8, shape.width * shape.height);
        }
        input_type = pixmask::PixelType::U8_RGB;
    } else if (is_float32) {
        if (shape.channels != 3) {
            throw py::value_error("float32 inputs must have exactly 3 channels");
        }
        storage_f32 = copy_buffer<float>(info, shape, shape.channels);
        input_type = pixmask::PixelType::F32_RGB;
    } else {
        throw py::value_error("unsupported dtype: expected uint8 or float32");
    }

    pixmask::PixelType output_type = pixmask::PixelType::U8_RGB;
    if (output_pref == OutputType::Auto) {
        output_type = is_float32 ? pixmask::PixelType::F32_RGB : pixmask::PixelType::U8_RGB;
    } else if (output_pref == OutputType::Uint8) {
        output_type = pixmask::PixelType::U8_RGB;
    } else {
        output_type = pixmask::PixelType::F32_RGB;
    }

    const std::size_t pixel_count = shape.width * shape.height;
    std::vector<std::uint8_t> output_u8;
    std::vector<float> output_f32;

    const std::size_t input_stride_bytes = shape.width * pixmask::bytes_per_pixel(input_type);
    const std::size_t output_stride_bytes = shape.width * pixmask::bytes_per_pixel(output_type);

    pixmask::CpuImage input_view;
    pixmask::CpuImage output_view;

    if (input_type == pixmask::PixelType::U8_RGB) {
        input_view = pixmask::CpuImage(input_type, shape.width, shape.height, input_stride_bytes, storage_u8.data());
    } else {
        input_view = pixmask::CpuImage(input_type, shape.width, shape.height, input_stride_bytes, storage_f32.data());
    }

    if (output_type == pixmask::PixelType::U8_RGB) {
        output_u8.assign(pixel_count * 3u, 0);
        output_view = pixmask::CpuImage(output_type, shape.width, shape.height, output_stride_bytes, output_u8.data());
    } else {
        output_f32.assign(pixel_count * 3u, 0.0f);
        output_view = pixmask::CpuImage(output_type, shape.width, shape.height, output_stride_bytes, output_f32.data());
    }

    if (!pixmask::sanitize(input_view, output_view)) {
        throw py::value_error("pixmask::sanitize returned failure");
    }

    if (output_type == pixmask::PixelType::U8_RGB) {
        py::array result = create_output_array(py::dtype::of<std::uint8_t>(), shape);
        std::memcpy(result.mutable_data(), output_u8.data(), output_u8.size() * sizeof(std::uint8_t));
        return result;
    }

    py::array result = create_output_array(py::dtype::of<float>(), shape);
    std::memcpy(result.mutable_data(), output_f32.data(), output_f32.size() * sizeof(float));
    return result;
}

} // namespace

PYBIND11_MODULE(_pixmask, m) {
    m.doc() = "Python bindings for the pixmask sanitization pipeline";

    pixmask::initialize();

    m.def("version", []() { return pixmask::version_string(); }, "Return the pixmask library version string");
    m.def("sanitize",
          &sanitize_array,
          py::arg("image"),
          "Run the pixmask sanitize pipeline on the provided image array");

    auto security = m.def_submodule("security", "Security utility helpers");
    security.def("exceeds_pixel_cap",
                 &pixmask::exceeds_pixel_cap,
                 py::arg("width"),
                 py::arg("height"),
                 py::arg("cap_megapixels"),
                 "Return True if an image exceeds the configured megapixel cap");

    security.def(
        "suspicious_polyglot_bytes",
        [](py::object obj) {
            py::object builtins = py::module_::import("builtins");
            const py::bytes data = builtins.attr("bytes")(obj);
            const std::string buffer = data;
            return pixmask::suspicious_polyglot_bytes(
                reinterpret_cast<const std::uint8_t *>(buffer.data()), buffer.size());
        },
        py::arg("buffer"),
        "Return True if a byte buffer contains suspicious polyglot signatures");
}

