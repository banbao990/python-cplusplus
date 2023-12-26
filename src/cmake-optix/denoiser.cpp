#include "denoiser.h"

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
    if (OPTIX_VERSION < 80000) {
        MI_LOG_E("Optix Should >= %d, yours is %d\n", 80000, OPTIX_VERSION);
        exit(1);
    }

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

void Denoiser::resize_denoised_img() {
    const int width = m_size.x;
    const int height = m_size.y;
    const int element_size = m_size.z;
    auto tensor_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0).requires_grad(false);
    m_denoised_img = torch::empty({height, width, int(element_size / sizeof(float))}, tensor_options);
}

void Denoiser::resize_optix_denoiser() {
    if (m_denoiser != nullptr) {
        OPTIX_CHECK(optixDenoiserDestroy(m_denoiser));
    }

    const int width = m_size.x;
    const int height = m_size.y;
    const int element_size = m_size.z;

    // ------------------------------------------------------------------
    // create the denoiser:
    OptixDenoiserOptions denoiser_options = {};
    if (m_aux) {
        denoiser_options.guideNormal = 1;
        denoiser_options.guideAlbedo = 1;
    }
    OPTIX_CHECK(optixDenoiserCreate(m_optix_context, OPTIX_DENOISER_MODEL_KIND_HDR, &denoiser_options, &m_denoiser));
    // .. then compute and allocate memory resources for the denoiser
    OptixDenoiserSizes denoiser_return_sizes;
    OPTIX_CHECK(optixDenoiserComputeMemoryResources(m_denoiser, width, height,
                                                    &denoiser_return_sizes));

    const int denoiser_scratch_size = (std::max)(denoiser_return_sizes.withOverlapScratchSizeInBytes,
                                                 denoiser_return_sizes.withoutOverlapScratchSizeInBytes);
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

void Denoiser::resize(const int3 size, const bool aux, const bool temporal) {

    const bool change_buffer = (size != m_size);
    const bool change_setting = (m_size.x != size.x) || (m_size.y != size.y) || (m_temporal != temporal) || (m_aux != aux);

    // update
    m_size = size;
    m_temporal = temporal;
    m_aux = aux;

    // output denoised image
    if (change_buffer) {
        resize_denoised_img();
    }

    // optix setup
    if (change_setting) {
        resize_optix_denoiser();
    }
}

torch::Tensor Denoiser::denoise(const torch::Tensor *img_with_noise,
                                const torch::Tensor *albedo,
                                const torch::Tensor *normal) {
    // guard
    if (m_aux) {
        assert(albedo != nullptr);
        assert(normal != nullptr);
    }

    const int height = img_with_noise->size(0);
    const int width = img_with_noise->size(1);
    const int element_size = img_with_noise->size(2) * sizeof(float);

    // TODO: const vars for now
    const bool accumulate = false;
    const int frame_id = 10;

    OptixDenoiserParams denoiserParams = {};
    denoiserParams.hdrIntensity = (CUdeviceptr)0;
    if (accumulate) {
        denoiserParams.blendFactor = 1.f / frame_id;
    } else {
        denoiserParams.blendFactor = 0.0f;
    }

    // -------------------------------------------------------
    OptixImage2D inputLayer = {};
    inputLayer.data = (CUdeviceptr)img_with_noise->data_ptr();
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

    OptixDenoiserGuideLayer denoiserGuideLayer = {};

    OptixDenoiserLayer denoiserLayer = {};
    denoiserLayer.input = inputLayer;
    denoiserLayer.output = outputLayer;
    if (m_aux) {
        inputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT3;
        inputLayer.data = (CUdeviceptr)albedo->data_ptr();
        denoiserGuideLayer.albedo = inputLayer;
        inputLayer.data = (CUdeviceptr)normal->data_ptr();
        denoiserGuideLayer.normal = inputLayer;
    }

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
    return m_denoised_img;
}

void Denoiser::context_log_cb(unsigned int level,
                              const char *tag,
                              const char *message,
                              void *) {
    MI_LOG("[%2d][%12s]: %s\n", (int)level, tag, message);
}
