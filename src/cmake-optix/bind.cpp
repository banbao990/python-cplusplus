// TODO: you should include the torch header first
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <pybind11/pybind11.h>

#include "denoiser.h"
// this include may only appear in a single source file:
#include <optix_function_table_definition.h>

#ifdef _WIN32
#include <windows.h>
LONG WINAPI MyUnhandledExceptionFilter(struct _EXCEPTION_POINTERS *ExceptionInfo) {
    if (ExceptionInfo->ExceptionRecord->ExceptionCode == EXCEPTION_ACCESS_VIOLATION) {
        MI_LOG_ERROR("Access violation caught, exiting gracefully\n");
        exit(1);
    }
    return EXCEPTION_EXECUTE_HANDLER;
}
#endif

void init_denoiser() {
    static bool is_init = false;
    if (is_init) {
        return;
    }

    if (is_init == false) {
#ifdef _WIN32
        SetUnhandledExceptionFilter(MyUnhandledExceptionFilter);
#endif
        is_init = true;
    }
}

torch::Tensor optix_denoise(const torch::Tensor &img_with_noise) {
    init_denoiser();
    const int height = img_with_noise.size(0);
    const int width = img_with_noise.size(1);
    const int element_size = img_with_noise.size(2) * sizeof(float);

    assert(element_size == sizeof(float3) || element_size == sizeof(float4));

    Denoiser *denoiser = Denoiser::get_instance();
    denoiser->resize({width, height, element_size});
    return denoiser->denoise(&img_with_noise, nullptr, nullptr);
}

torch::Tensor optix_denoise_aux(const torch::Tensor &img_with_noise,
                                const torch::Tensor &albedo,
                                const torch::Tensor &normal) {
    init_denoiser();
    const int height = img_with_noise.size(0);
    const int width = img_with_noise.size(1);
    const int element_size = img_with_noise.size(2) * sizeof(float);

    assert(element_size == sizeof(float3) || element_size == sizeof(float4));

    Denoiser *denoiser = Denoiser::get_instance();
    denoiser->resize({width, height, element_size}, true);
    return denoiser->denoise(&img_with_noise, &albedo, &normal);
}

PYBIND11_MODULE(cmake_optix_example, m) {
    m.def("init", &init_denoiser, "init the denoiser");
    m.def("denoise",
          &optix_denoise,
          "denoise the output color");
    m.def("denoise_aux", &optix_denoise_aux, "denoise the output color with albedo and normal");
}
