#ifndef GPU_RELU_ACTIVATEFUNC_H
#define GPU_RELU_ACTIVATEFUNC_H

#include <gpu_based_layer.h>

namespace gpu_cnn{
class gpu_relu_activatefunc : public gpu_based_layer
{
    public:
        gpu_relu_activatefunc(cudnnHandle_t &_cudnnHandle, cudnnDataType_t &_dataType,cudnnTensorFormat_t &_tensorFormat)
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

            //Perform ReLU Activation Function
            checkCUDNN( cudnnActivationForward(cudnnHandle,CUDNN_ACTIVATION_RELU,&alpha_relu,srcTensorDesc,**input_d,&beta_relu,dstTensorDesc,**output_d));
        }
    private:
        cudnnHandle_t cudnnHandle;
        cudnnDataType_t dataType;
        cudnnTensorFormat_t tensorFormat;

        cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;

        float alpha_relu = 1;
        float beta_relu = 0;
};
}

#endif // GPU_RELU_ACTIVATEFUNC_H
