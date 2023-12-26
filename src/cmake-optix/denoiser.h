#pragma once
// TODO: you should include the torch header first
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>

#include "optix_helper.h"
#include <iostream>

class Denoiser {
private:
    bool m_initialized = false;

    // CUstream m_stream = nullptr;
    cudaDeviceProp m_device_props = {};
    CUcontext m_cuda_context = nullptr;
    OptixDeviceContext m_optix_context = nullptr;

    OptixDenoiser m_denoiser = nullptr;
    torch::Tensor m_denoiser_scratch = {};
    torch::Tensor m_denoiser_state = {};
    torch::Tensor m_denoised_img = {};
    int3 m_size = {0, 0, 0};

private:
    Denoiser();

public:
    // Delete copy constructor and assignment operator
    Denoiser(const Denoiser &) = delete;
    Denoiser &operator=(const Denoiser &) = delete;

    // Singleton
    static Denoiser *get_instance();
    ~Denoiser();
    void init();
    void resize(const int3 size);
    void resize_denoised_img(const int3 size);
    void resize_optix_denoiser(const int3 size);

    torch::Tensor denoise(const torch::Tensor &img_with_noise);

    static void context_log_cb(unsigned int level,
                               const char *tag,
                               const char *message,
                               void *);
};
