// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

// optix 7
#include <optix.h>
#include <optix_stubs.h>
#include <sstream>
#include <stdexcept>
#include <vector_types.h>
#include <cuda_runtime.h>

#ifdef __INTELLISENSE__
// just for MSVS's intelligence
#define __CUDACC__
#endif

#define CUDA_CHECK(call)                                   \
    {                                                      \
        cudaError_t rc = call;                             \
        if (rc != cudaSuccess) {                           \
            std::stringstream txt;                         \
            cudaError_t err = rc; /*cudaGetLastError();*/  \
            txt << "CUDA Error " << cudaGetErrorName(err)  \
                << " (" << cudaGetErrorString(err) << ")"; \
            throw std::runtime_error(txt.str());           \
        }                                                  \
    }

#define CUDA_CHECK_NOEXCEPT(call) \
    {                             \
        call;                     \
    }

#define OPTIX_CHECK(call)                                                                             \
    {                                                                                                 \
        OptixResult res = call;                                                                       \
        if (res != OPTIX_SUCCESS) {                                                                   \
            fprintf(stderr, "Optix call (%s) failed with code %d (line %d)\n", #call, res, __LINE__); \
            exit(2);                                                                                  \
        }                                                                                             \
    }

#define CUDA_SYNC_CHECK()                                                                                \
    {                                                                                                    \
        cudaDeviceSynchronize();                                                                         \
        cudaError_t error = cudaGetLastError();                                                          \
        if (error != cudaSuccess) {                                                                      \
            fprintf(stderr, "error (%s: line %d): %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(2);                                                                                     \
        }                                                                                                \
    }

#define MI_LOG_ERROR(msg) fprintf(stderr, "[Error] %s\n", msg);
#define MI_LOG_INFO(msg) fprintf(stdout, "[Info] %s\n", msg);

#define MI_LOG(format, ...) fprintf(stdout, format, __VA_ARGS__)
#define MI_LOG_E(format, ...) fprintf(stderr, format, __VA_ARGS__)

#define MI_OUTPUT_LINE fprintf(stderr, "[Info] lines: %d\n", __LINE__);

// math
bool operator==(const int3 &a, const int3 &b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

bool operator!=(const int3 &a, const int3 &b) {
    return !(a == b);
}
