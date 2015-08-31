#ifndef GPU_SOFTMAX_ACTIVATEFUNC_H
#define GPU_SOFTMAX_ACTIVATEFUNC_H

#include <gpu_based_layer.h>

namespace gpu_cnn{
class gpu_softmax_activatefunc : public gpu_based_layer
{
    public:
        gpu_softmax_activatefunc(cudnnHandle_t &_cudnnHandle, cudnnDataType_t &_dataType,cudnnTensorFormat_t &_tensorFormat)
        {
            cudnnHandle = _cudnnHandle;
            dataType = _dataType;
            tensorFormat = _tensorFormat;

            checkCUDNN( cudnnCreateTensorDescriptor(&srcTensorDesc) );
            checkCUDNN( cudnnCreateTensorDescriptor(&dstTensorDesc) );
        }

        void cleanup()
        {
            checkCUDNN( cudnnDestroyTensorDescriptor(srcTensorDesc));
            checkCUDNN( cudnnDestroyTensorDescriptor(dstTensorDesc));
        }

        void feed_forward(float*** input_d,int& n,int& c,int& h,int& w,float*** output_d)
        {
            checkCUDNN( cudnnSetTensor4dDescriptor(srcTensorDesc,tensorFormat,dataType,n,c,h,w));
            checkCUDNN( cudnnSetTensor4dDescriptor(dstTensorDesc,tensorFormat,dataType,n,c,h,w));

            //Allocate for output
            checkCudaErrors( cudaMalloc(&**output_d,n*c*h*w*sizeof(float)));

            checkCUDNN( cudnnSoftmaxForward(cudnnHandle,CUDNN_SOFTMAX_FAST,CUDNN_SOFTMAX_MODE_CHANNEL,&alpha_softmax,srcTensorDesc,**input_d,&beta_softmax,dstTensorDesc,**output_d));
        }

    private:
        cudnnHandle_t cudnnHandle;
        cudnnDataType_t dataType;
        cudnnTensorFormat_t tensorFormat;

        cudnnTensorDescriptor_t srcTensorDesc,dstTensorDesc;

        float alpha_softmax = 1;
        float beta_softmax = 0;
};
}
#endif // GPU_SOFTMAX_ACTIVATEFUNC_H
