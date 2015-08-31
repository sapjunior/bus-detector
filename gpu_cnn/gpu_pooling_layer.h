#ifndef GPU_POOLING_LAYER_H
#define GPU_POOLING_LAYER_H

#include <gpu_based_layer.h>

namespace gpu_cnn{
class gpu_pooling_layer : public gpu_based_layer
{
    public:
        gpu_pooling_layer(cudnnHandle_t &_cudnnHandle, cudnnDataType_t &_dataType,cudnnTensorFormat_t &_tensorFormat,
                            int _poolX,int _poolY,int strideX,int strideY)
        {
            cudnnHandle = _cudnnHandle;
            dataType = _dataType;
            tensorFormat = _tensorFormat;
            poolX = _poolX;
            poolY = _poolY;

            checkCUDNN( cudnnCreateTensorDescriptor(&srcTensorDesc));
            checkCUDNN( cudnnCreateTensorDescriptor(&dstTensorDesc));
            checkCUDNN( cudnnCreatePoolingDescriptor(&poolingDesc));

            checkCUDNN( cudnnSetPooling2dDescriptor(poolingDesc,CUDNN_POOLING_MAX,poolX,poolY,0,0,strideX,strideY));
        }

        void cleanup()
        {
            checkCUDNN( cudnnDestroyTensorDescriptor(srcTensorDesc));
            checkCUDNN( cudnnDestroyTensorDescriptor(dstTensorDesc));
            checkCUDNN( cudnnDestroyPoolingDescriptor(poolingDesc));
        }

        void feed_forward(float*** input_d,int& n,int& c,int& h,int& w,float*** output_d)
        {
            checkCUDNN( cudnnSetTensor4dDescriptor(srcTensorDesc,tensorFormat,dataType,n,c,h,w));

            h = h/poolX;
            w = w/poolY;
            checkCUDNN( cudnnSetTensor4dDescriptor(dstTensorDesc,tensorFormat,dataType,n,c,h,w));

            //Allocate for output
            checkCudaErrors( cudaMalloc(&**output_d,n*c*h*w*sizeof(float)));

            checkCUDNN( cudnnPoolingForward(cudnnHandle,poolingDesc,&alpha_pooling,srcTensorDesc,**input_d,&beta_pooling,dstTensorDesc,**output_d));
        }

    private:
        cudnnHandle_t cudnnHandle;
        cudnnDataType_t dataType;
        cudnnTensorFormat_t tensorFormat;

        cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;

        cudnnPoolingDescriptor_t poolingDesc;
        int poolX,poolY;
        float alpha_pooling = 1;
        float beta_pooling = 0;
};
}
#endif // GPU_POOLING_LAYER_H
