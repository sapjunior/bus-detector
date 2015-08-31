#ifndef GPU_BASED_LAYER_H
#define GPU_BASED_LAYER_H

#include <iostream>
#include <sstream>
#include <stdlib.h>
#include "cuda.h"
#include "cudnn.h"
#include "cublas_v2.h"

//Error Checking Function
#define FatalError(s) {                                                \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;\
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(EXIT_FAILURE);                                                \
}
#define checkCUDNN(status) {                                           \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
      FatalError(_error.str());                                        \
    }                                                                  \
}

#define checkCudaErrors(status) {                                      \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << cudaGetErrorString(status);        \
      FatalError(_error.str());                                        \
    }                                                                  \
}

namespace gpu_cnn
{
class gpu_based_layer
{
public:
    gpu_based_layer() {}
    virtual void feed_forward(float*** input_d,int& n,int& c,int& h,int& w,float*** output_d) = 0;
    virtual void cleanup() = 0;

};
}
#endif // GPU_BASED_LAYER_H
