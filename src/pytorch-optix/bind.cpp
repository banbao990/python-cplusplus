#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>
#include "common.h"
#include "optix-helper.h"
#include "optix_function_table_definition.h"

#include <iostream>

#ifdef _WIN32
#include <windows.h>
LONG WINAPI MyUnhandledExceptionFilter(struct _EXCEPTION_POINTERS *ExceptionInfo) {
    if (ExceptionInfo->ExceptionRecord->ExceptionCode == EXCEPTION_ACCESS_VIOLATION) {
        std::cout << "Access violation caught, exiting gracefully\n";
        exit(1);
    }
    return EXCEPTION_EXECUTE_HANDLER;
}
#endif

// single instance of optix denoiser
static CUstream s_stream = nullptr;
static cudaDeviceProp s_device_props = {};
static CUcontext s_cuda_context = nullptr;
static OptixDeviceContext s_optix_context = nullptr;
static OptixDenoiser s_denoiser = nullptr;

static torch::Tensor s_denoiser_scratch = {};
static torch::Tensor s_denoiser_state = {};
static torch::Tensor s_denoised_img = {};

static void context_log_cb(unsigned int level,
                           const char *tag,
                           const char *message,
                           void *) {
    fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
}

OptixDenoiser create_optix_denoiser(const int width, const int height, const int element_size) {

    if (s_optix_context == nullptr) {
#ifdef _WIN32
        SetUnhandledExceptionFilter(MyUnhandledExceptionFilter);
#endif
        // init optix
        cudaFree(0);
        int numDevices;
        cudaGetDeviceCount(&numDevices);
        if (numDevices == 0)
            throw std::runtime_error("[ERROR]: no CUDA capable devices found!");
        OPTIX_CHECK(optixInit());

        // create context
        const int device_id = 0;
        CUDA_CHECK(cudaSetDevice(device_id));
        CUDA_CHECK(cudaStreamCreate(&s_stream));

        cudaGetDeviceProperties(&s_device_props, device_id);
        std::cout
            << "[Info] optix version: " << OPTIX_VERSION << std::endl
            << "[Info] cuda version: " << CUDART_VERSION << std::endl
            << "[optix] device: " << s_device_props.name << std::endl;

        CUresult cu_res = cuCtxGetCurrent(&s_cuda_context);
        if (cu_res != CUDA_SUCCESS)
            fprintf(stderr, "Error querying current context: error code %d\n", cu_res);

        OPTIX_CHECK(optixDeviceContextCreate(s_cuda_context, 0, &s_optix_context));
        OPTIX_CHECK(optixDeviceContextSetLogCallback(s_optix_context, context_log_cb, nullptr, 4));
    }

    // ------------------------------------------------------------------
    // create the denoiser:
    OptixDenoiserOptions denoiser_options = {};

#if OPTIX_VERSION >= 70300
    OPTIX_CHECK(optixDenoiserCreate(s_optix_context, OPTIX_DENOISER_MODEL_KIND_HDR, &denoiser_options, &s_denoiser));
#else
    denoiser_options.inputKind = OPTIX_DENOISER_INPUT_RGB;

#if OPTIX_VERSION < 70100
    // these only exist in 7.0, not 7.1
    denoiser_options.pixelFormat = OPTIX_PIXEL_FORMAT_FLOAT4;
#endif

    OPTIX_CHECK(optixDenoiserCreate(s_optix_context, &denoiser_options, &s_denoiser));
    OPTIX_CHECK(optixDenoiserSetModel(s_denoiser, OPTIX_DENOISER_MODEL_KIND_HDR, NULL, 0));
#endif

    // .. then compute and allocate memory resources for the denoiser
    OptixDenoiserSizes denoiser_return_sizes;
    OPTIX_CHECK(optixDenoiserComputeMemoryResources(s_denoiser, width, height,
                                                    &denoiser_return_sizes));
    std::cout << std::endl
              << "stateSizeInBytes: " << denoiser_return_sizes.stateSizeInBytes << std::endl
              << "withOverlapScratchSizeInBytes: " << denoiser_return_sizes.withOverlapScratchSizeInBytes << std::endl
              << "withoutOverlapScratchSizeInBytes: " << denoiser_return_sizes.withoutOverlapScratchSizeInBytes << std::endl
              << "overlapWindowSizeInPixels: " << denoiser_return_sizes.overlapWindowSizeInPixels << std::endl
              << "computeAverageColorSizeInBytes: " << denoiser_return_sizes.computeAverageColorSizeInBytes << std::endl
              << "computeIntensitySizeInBytes: " << denoiser_return_sizes.computeIntensitySizeInBytes << std::endl
              << "internalGuideLayerPixelSizeInBytes: " << denoiser_return_sizes.internalGuideLayerPixelSizeInBytes << std::endl
              << std::endl;

#if OPTIX_VERSION < 70100
    const int denoiser_scratch_size = denoiser_return_sizes.recommendedScratchSizeInBytes;
#else
    const int denoiser_scratch_size = (std::max)(denoiser_return_sizes.withOverlapScratchSizeInBytes,
                                                 denoiser_return_sizes.withoutOverlapScratchSizeInBytes);
#endif
    auto tensor_options = torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA, 0).requires_grad(false);
    s_denoiser_scratch = torch::empty(denoiser_scratch_size, tensor_options);

    const int denoiser_state_size = denoiser_return_sizes.stateSizeInBytes;
    s_denoiser_state = torch::empty(denoiser_state_size, tensor_options);

    // ------------------------------------------------------------------
    // resize our cuda frame buffer
    tensor_options = tensor_options.dtype(torch::kFloat32);
    s_denoised_img = torch::empty({height, width, int(element_size / sizeof(float))}, tensor_options);

    std::cout << std::endl
              << "width: " << width << std::endl
              << "height: " << height << std::endl
              << "element_size: " << element_size << std::endl
              << "denoiser_scratch_size: " << denoiser_scratch_size << std::endl
              << "denoiser_scratch_size(tensor): " << s_denoiser_scratch.sizes()[0] << std::endl
              << "denoiser_state_size: " << denoiser_state_size << std::endl
              << "denoiser_state_size(tensor): " << s_denoiser_state.sizes()[0] << std::endl
              << std::endl;

    // ------------------------------------------------------------------
    OPTIX_CHECK(
        optixDenoiserSetup(s_denoiser, 0,
                           width, height,
                           (CUdeviceptr)s_denoiser_state.data_ptr(),
                           s_denoiser_state.sizes()[0],
                           (CUdeviceptr)s_denoiser_scratch.data_ptr(),
                           s_denoiser_scratch.sizes()[0]));

    return s_denoiser;
}

torch::Tensor optix_denoiser(const torch::Tensor &img_with_noise, const torch::Tensor &b) {
    const int height = img_with_noise.size(0);
    const int width = img_with_noise.size(1);
    const int element_size = img_with_noise.size(2) * sizeof(float);

    // FIXME: we assume the element size won't change in the lifetime of the denoiser
    assert(element_size == sizeof(float3) || element_size == sizeof(float4));

    if (s_denoiser == nullptr) {
        s_denoiser = create_optix_denoiser(width, height, element_size);
    }

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
    outputLayer.data = (CUdeviceptr)s_denoised_img.data_ptr();

#if OPTIX_VERSION >= 70300
    OptixDenoiserGuideLayer denoiserGuideLayer = {};

    OptixDenoiserLayer denoiserLayer = {};
    denoiserLayer.input = inputLayer;
    denoiserLayer.output = outputLayer;

    OPTIX_CHECK(optixDenoiserInvoke(s_denoiser,
                                    /*stream*/ 0,
                                    &denoiserParams,
                                    (CUdeviceptr)s_denoiser_state.data_ptr(),
                                    s_denoiser_state.sizes()[0],
                                    &denoiserGuideLayer,
                                    &denoiserLayer, 1,
                                    /*inputOffsetX*/ 0,
                                    /*inputOffsetY*/ 0,
                                    (CUdeviceptr)s_denoiser_scratch.data_ptr(),
                                    s_denoiser_scratch.sizes()[0]));
#else
    OPTIX_CHECK(optixDenoiserInvoke(denoiser,
                                    /*stream*/ 0,
                                    &denoiserParams,
                                    s_denoiser_state.data_ptr(),
                                    s_denoiser_state_size,
                                    &inputLayer, 1,
                                    /*inputOffsetX*/ 0,
                                    /*inputOffsetY*/ 0,
                                    &outputLayer,
                                    s_denoiser_scratch.data_ptr(),
                                    s_denoiser_scratch_size));
#endif
    return s_denoised_img;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("optix_denoiser",
          &optix_denoiser,
          "example for pytorch optix extension");
}
