#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>

#include "oidn_denoiser.h"
#include <pybind11/pybind11.h>
#include <iostream>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

void init() {
    OidnDenoiser *oidn_denoiser = OidnDenoiser::get_instance();
    std::cout << "OidnDenoiser initialized successfully" << std::endl;
}

void denoise(torch::Tensor color, torch::Tensor output, int width, int height, int channels) {
    if (channels > 3 || channels < 1) {
        std::cerr << "[Error] Oidn only supports up to 3 channels" << std::endl;
        return;
    }

    OidnDenoiser *oidn_denoiser = OidnDenoiser::get_instance();
    oidn_denoiser->denoise((float *)color.data_ptr(), (float *)output.data_ptr(), width, height, channels);
}

PYBIND11_MODULE(oidn_example, m) {
    m.doc() = "pybind11 oidn plugin";  // optional module docstring
    m.def("init", &init, "Initialize oidn denoiser");
    m.def("denoise", &denoise, "Denoise image");
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}