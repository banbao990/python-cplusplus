// TODO: you should include the torch header first
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>
#include "denoiser.hpp"

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

static bool s_initialized = false;

torch::Tensor optix_denoise(const torch::Tensor &img_with_noise) {
    const int height = img_with_noise.size(0);
    const int width = img_with_noise.size(1);
    const int element_size = img_with_noise.size(2) * sizeof(float);

    assert(element_size == sizeof(float3) || element_size == sizeof(float4));

    if (s_initialized == false) {
#ifdef _WIN32
        SetUnhandledExceptionFilter(MyUnhandledExceptionFilter);
#endif
        s_initialized = true;
    }

    Denoiser *denoiser = Denoiser::get_instance();
    denoiser->resize({width, height, element_size});
    return denoiser->denoise(img_with_noise);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("optix_denoise",
          &optix_denoise,
          "optix denoiser example");
}
