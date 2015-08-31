#ifndef GPU_PREPROCESS_LAYER_H
#define GPU_PREPROCESS_LAYER_H

#include "gpu_based_layer.h"

namespace gpu_cnn{
class gpu_preprocess_layer : public gpu_based_layer
{
    public:
        gpu_preprocess_layer(cudnnHandle_t &_cudnnHandle, cudnnDataType_t &_dataType,cudnnTensorFormat_t &_tensorFormat,
                            float* _averageImage_d)
        {
            cudnnHandle = _cudnnHandle;
            dataType = _dataType;
            tensorFormat = _tensorFormat;
            averageImage_d = _averageImage_d;

            checkCUDNN( cudnnCreateTensorDescriptor(&srcTensorDesc) );
            checkCUDNN( cudnnCreateTensorDescriptor(&dstTensorDesc) );
            checkCUDNN( cudnnCreateTensorDescriptor(&averageImageTensorDesc) );
        }

        void cleanup()
        {
            checkCUDNN( cudnnDestroyTensorDescriptor(srcTensorDesc));
            checkCUDNN( cudnnDestroyTensorDescriptor(dstTensorDesc));
            checkCUDNN( cudnnDestroyTensorDescriptor(averageImageTensorDesc));

            checkCudaErrors( cudaFree(averageImage_d));
        }

        void feed_forward(float*** input_d,int& n,int& c,int& h,int& w,float*** output_d)
        {
            //Output Tensor Descriptor
            checkCUDNN( cudnnSetTensor4dDescriptor(dstTensorDesc,tensorFormat,dataType,n,c,h,w));

            //Bias Tensor
            checkCUDNN( cudnnSetTensor4dDescriptor(averageImageTensorDesc,tensorFormat,dataType,1,3,h,w));

            **output_d=**input_d;
            **input_d=NULL;

            //Add Bias
            checkCUDNN( cudnnAddTensor(cudnnHandle,CUDNN_ADD_SAME_C,&alpha_add,averageImageTensorDesc,averageImage_d,&beta_add,dstTensorDesc,**output_d));
        }
    private:
        cudnnHandle_t cudnnHandle;
        cudnnDataType_t dataType;
        cudnnTensorFormat_t tensorFormat;

        cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc, averageImageTensorDesc;
        float *averageImage_d;

        float alpha_add = -1;
        float beta_add = 1;
};
}
#endif // GPU_PREPROCESS_LAYER_H
