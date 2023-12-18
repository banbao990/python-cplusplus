#pragma once
// TODO: you should include the torch header first
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>

#include "optix-helper.h"
#include "optix_function_table_definition.h"
#include <iostream>

///////////////////////////////////////////////////////////////////////////////
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

///////////////////////////////////////////////////////////////////////////////

Denoiser *Denoiser::get_instance() {
    static Denoiser s_instance;
    return &s_instance;
}

Denoiser::Denoiser() {
    init();
}

Denoiser::~Denoiser() {
    try {
        if (m_denoiser != nullptr) {
            OPTIX_CHECK(optixDenoiserDestroy(m_denoiser));
        }

        if (m_initialized) {
            OPTIX_CHECK(optixDeviceContextDestroy(m_optix_context));
            // CUDA_CHECK(cudaStreamDestroy(m_stream));
        }
    } catch (const std::exception &e) {
        MI_LOG_E("Error when destroying denoiser: %s\n", e.what());
    } catch (...) {
        MI_LOG_E("Error when destroying denoiser: unknown error\n");
    }
}

void Denoiser::init() {
    if (m_initialized) {
        return;
    }

    // init optix
    cudaFree(0);
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0)
        throw std::runtime_error("[ERROR]: no CUDA capable devices found!");
    OPTIX_CHECK(optixInit());

    // create context
    const int deviceID = 0;
    CUDA_CHECK(cudaSetDevice(deviceID));
    // CUDA_CHECK(cudaStreamCreate(&m_stream));

    cudaGetDeviceProperties(&m_device_props, deviceID);
    MI_LOG("[Info] optix version: %d\n", OPTIX_VERSION);
    MI_LOG("[Info] cuda version: %d\n", CUDART_VERSION);
    MI_LOG("[optix] device: %s\n", m_device_props.name);

    CUresult cuRes = cuCtxGetCurrent(&m_cuda_context);

    if (cuRes != CUDA_SUCCESS) {
        MI_LOG_E("Error querying current context: error code %d\n", cuRes);
    }

    OPTIX_CHECK(optixDeviceContextCreate(m_cuda_context, 0, &m_optix_context));
    OPTIX_CHECK(optixDeviceContextSetLogCallback(m_optix_context, context_log_cb, nullptr, 4));

    m_initialized = true;
}

void Denoiser::resize_denoised_img(const int3 size) {
    const int &width = size.x;
    const int &height = size.y;
    const int &element_size = size.z;

    auto tensor_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0).requires_grad(false);
    m_denoised_img = torch::empty({height, width, int(element_size / sizeof(float))}, tensor_options);
}

void Denoiser::resize_optix_denoiser(const int3 size) {
    if (m_denoiser != nullptr) {
        OPTIX_CHECK(optixDenoiserDestroy(m_denoiser));
    }

    const int &width = size.x;
    const int &height = size.y;
    const int &element_size = size.z;

    // ------------------------------------------------------------------
    // create the denoiser:
    OptixDenoiserOptions denoiser_options = {};

#if OPTIX_VERSION >= 70300
    OPTIX_CHECK(optixDenoiserCreate(m_optix_context, OPTIX_DENOISER_MODEL_KIND_HDR, &denoiser_options, &m_denoiser));
#else
    denoiser_options.inputKind = OPTIX_DENOISER_INPUT_RGB;

#if OPTIX_VERSION < 70100
    // these only exist in 7.0, not 7.1
    denoiser_options.pixelFormat = OPTIX_PIXEL_FORMAT_FLOAT4;
#endif

    OPTIX_CHECK(optixDenoiserCreate(m_optix_context, &denoiser_options, &m_denoiser));
    OPTIX_CHECK(optixDenoiserSetModel(m_denoiser, OPTIX_DENOISER_MODEL_KIND_HDR, NULL, 0));
#endif

    // .. then compute and allocate memory resources for the denoiser
    OptixDenoiserSizes denoiser_return_sizes;
    OPTIX_CHECK(optixDenoiserComputeMemoryResources(m_denoiser, width, height,
                                                    &denoiser_return_sizes));

#if OPTIX_VERSION < 70100
    const int denoiser_scratch_size = denoiser_return_sizes.recommendedScratchSizeInBytes;
#else
    const int denoiser_scratch_size = (std::max)(denoiser_return_sizes.withOverlapScratchSizeInBytes,
                                                 denoiser_return_sizes.withoutOverlapScratchSizeInBytes);
#endif
    auto tensor_options = torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA, 0).requires_grad(false);
    m_denoiser_scratch = torch::empty(denoiser_scratch_size, tensor_options);

    const int denoiser_state_size = denoiser_return_sizes.stateSizeInBytes;
    m_denoiser_state = torch::empty(denoiser_state_size, tensor_options);

    // ------------------------------------------------------------------
    OPTIX_CHECK(
        optixDenoiserSetup(m_denoiser, 0,
                           width, height,
                           (CUdeviceptr)m_denoiser_state.data_ptr(),
                           m_denoiser_state.sizes()[0],
                           (CUdeviceptr)m_denoiser_scratch.data_ptr(),
                           m_denoiser_scratch.sizes()[0]));
}

void Denoiser::resize(const int3 size) {
    // output denoised image
    if (size != m_size) {
        resize_denoised_img(size);
    }

    // optix setup
    if (!(size.x == m_size.x && size.y == m_size.y)) {
        resize_optix_denoiser(size);
    }

    m_size = size;
}

torch::Tensor Denoiser::denoise(const torch::Tensor &img_with_noise) {
    const int height = img_with_noise.size(0);
    const int width = img_with_noise.size(1);
    const int element_size = img_with_noise.size(2) * sizeof(float);

    // TODO: const vars for now
    const bool accumulate = false;
    const int frame_id = 10;

    OptixDenoiserParams denoiserParams = {};
#if OPTIX_VERSION >= 80000
#elif OPTIX_VERSION > 70500
    denoiserParams.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_ALPHA_AS_AOV;
#endif
    denoiserParams.hdrIntensity = (CUdeviceptr)0;
    if (accumulate)
        denoiserParams.blendFactor = 1.f / frame_id;
    else
        denoiserParams.blendFactor = 0.0f;

    // -------------------------------------------------------
    OptixImage2D inputLayer = {};
    inputLayer.data = (CUdeviceptr)img_with_noise.data_ptr();
    /// Width of the image (in pixels)
    inputLayer.width = width;
    /// Height of the image (in pixels)
    inputLayer.height = height;
    /// Stride between subsequent rows of the image (in bytes).
    inputLayer.rowStrideInBytes = width * element_size;
    /// Stride between subsequent pixels of the image (in bytes).
    /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
    inputLayer.pixelStrideInBytes = element_size;
    /// Pixel format.
    inputLayer.format = (element_size == sizeof(float4)) ? OPTIX_PIXEL_FORMAT_FLOAT4 : OPTIX_PIXEL_FORMAT_FLOAT3;

    // -------------------------------------------------------
    OptixImage2D outputLayer = inputLayer;
    outputLayer.data = (CUdeviceptr)m_denoised_img.data_ptr();

#if OPTIX_VERSION >= 70300
    OptixDenoiserGuideLayer denoiserGuideLayer = {};

    OptixDenoiserLayer denoiserLayer = {};
    denoiserLayer.input = inputLayer;
    denoiserLayer.output = outputLayer;

    OPTIX_CHECK(optixDenoiserInvoke(m_denoiser,
                                    /*stream*/ 0,
                                    &denoiserParams,
                                    (CUdeviceptr)m_denoiser_state.data_ptr(),
                                    m_denoiser_state.sizes()[0],
                                    &denoiserGuideLayer,
                                    &denoiserLayer, 1,
                                    /*inputOffsetX*/ 0,
                                    /*inputOffsetY*/ 0,
                                    (CUdeviceptr)m_denoiser_scratch.data_ptr(),
                                    m_denoiser_scratch.sizes()[0]));
#else
    OPTIX_CHECK(optixDenoiserInvoke(denoiser,
                                    /*stream*/ 0,
                                    &denoiserParams,
                                    m_denoiser_state.data_ptr(),
                                    s_denoiser_state_size,
                                    &inputLayer, 1,
                                    /*inputOffsetX*/ 0,
                                    /*inputOffsetY*/ 0,
                                    &outputLayer,
                                    m_denoiser_scratch.data_ptr(),
                                    s_denoiser_scratch_size));
#endif
    // MI_LOG("%lu\n", (unsigned long)img_with_noise.data_ptr());
    return m_denoised_img;
}

void Denoiser::context_log_cb(unsigned int level,
                              const char *tag,
                              const char *message,
                              void *) {
    MI_LOG_E("[%2d][%12s]: %s\n", (int)level, tag, message);
}

///////////////////////////////////////////////////////////////////////////////