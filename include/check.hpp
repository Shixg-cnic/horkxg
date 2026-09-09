#pragma once

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t error__ = (call);                                      \
        if (error__ != cudaSuccess) {                                            \
            std::ostringstream stream__;                                         \
            stream__ << "CUDA failure at " << __FILE__ << ':' << __LINE__       \
                     << " for " << #call << ": "                                \
                     << cudaGetErrorString(error__);                             \
            throw std::runtime_error(stream__.str());                            \
        }                                                                       \
    } while (0)
