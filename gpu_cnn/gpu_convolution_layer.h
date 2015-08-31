#ifndef GPU_CONVOLUTION_LAYER_H
#define GPU_CONVOLUTION_LAYER_H

#include <gpu_based_layer.h>

namespace gpu_cnn
{
class gpu_convolution_layer : public gpu_based_layer
{
public:
    gpu_convolution_layer(cudnnHandle_t &_cudnnHandle, cudnnDataType_t &_dataType,cudnnTensorFormat_t &_tensorFormat,
                          int _Sz,int _Sd,int _Sx,int _Sy,float _pad,
                          float* _filter_d,float* _bias_d)
    {
        cudnnHandle = _cudnnHandle;
        dataType = _dataType;
        tensorFormat = _tensorFormat;
        Sz = _Sz;
        Sd = _Sd;
        Sx = _Sx;
        Sy = _Sy;
        padH = _pad;
        padW = _pad;
        filter_d = _filter_d;
        bias_d = _bias_d;

        checkCUDNN( cudnnCreateTensorDescriptor(&srcTensorDesc) );
        checkCUDNN( cudnnCreateTensorDescriptor(&dstTensorDesc) );
        checkCUDNN( cudnnCreateTensorDescriptor(&biasTensorDesc) );
        checkCUDNN( cudnnCreateFilterDescriptor(&filterDesc) );
        checkCUDNN( cudnnCreateConvolutionDescriptor(&convDesc) );

        //Filter Descriptor                                      //output F - input F - kernelW - kernelH
        checkCUDNN( cudnnSetFilter4dDescriptor(filterDesc,dataType,Sd,Sz,Sx,Sy));

        //Convolution Descriptor                                //padH - padW - stridesY - stridesX - 1 - 1
        checkCUDNN( cudnnSetConvolution2dDescriptor(convDesc,padH,padW,1,1,1,1,CUDNN_CROSS_CORRELATION));
    }

    void cleanup()
    {
        checkCUDNN( cudnnDestroyTensorDescriptor(srcTensorDesc));
        checkCUDNN( cudnnDestroyTensorDescriptor(dstTensorDesc));
        checkCUDNN( cudnnDestroyTensorDescriptor(biasTensorDesc));
        checkCUDNN( cudnnDestroyConvolutionDescriptor(convDesc));
        checkCUDNN( cudnnDestroyFilterDescriptor(filterDesc));
        checkCudaErrors( cudaFree(filter_d));
        checkCudaErrors( cudaFree(bias_d));
    }

    void feed_forward(float*** input_d,int& n,int& c,int& h,int& w,float*** output_d)
    {
        //Input Tensor Descriptor
        checkCUDNN( cudnnSetTensor4dDescriptor(srcTensorDesc,tensorFormat,dataType,n,c,h,w));

        //Convolution Output Dims [feed n c h w as address to internally MODIFY them]
        checkCUDNN( cudnnGetConvolution2dForwardOutputDim(convDesc,srcTensorDesc,filterDesc,&n,&c,&h,&w));

        //Output Tensor Descriptor
        checkCUDNN( cudnnSetTensor4dDescriptor(dstTensorDesc,tensorFormat,dataType,n,c,h,w));

        //Convolve Algorithm (save GPU memory mode)
        checkCUDNN( cudnnGetConvolutionForwardAlgorithm(cudnnHandle,srcTensorDesc,filterDesc,convDesc,dstTensorDesc,CUDNN_CONVOLUTION_FWD_NO_WORKSPACE,0,&algo));

        //Workspace Setup
//      size_t sizeInBytes = 0;
//      void* workSpace = NULL;
//      checkCUDNN( cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,srcTensorDesc,filterDesc,convDesc,dstTensorDesc,algo,&sizeInBytes));
/*
        //Allocate Memory for Convolution
        if (sizeInBytes!=0)
        {
            checkCudaErrors( cudaMalloc(&workSpace,sizeInBytes) );
        }
*/
        checkCudaErrors( cudaMalloc(&**output_d,n*c*h*w*sizeof(float)));
        checkCUDNN( cudnnConvolutionForward(cudnnHandle,&alpha_conv,srcTensorDesc,**input_d,filterDesc,filter_d,convDesc,algo,NULL,0,&beta_conv,dstTensorDesc,**output_d) );
/*
        if (sizeInBytes!=0)
        {
            checkCudaErrors( cudaFree(workSpace));
        }
*/
        //Bias Tensor
        checkCUDNN( cudnnSetTensor4dDescriptor(biasTensorDesc,tensorFormat,dataType,1,c,1,1));

        //Add Bias
        checkCUDNN( cudnnAddTensor(cudnnHandle,CUDNN_ADD_SAME_C,&alpha_add,biasTensorDesc,bias_d,&beta_add,dstTensorDesc,**output_d));
    }
private:
    cudnnConvolutionFwdAlgo_t algo;

    cudnnHandle_t cudnnHandle;
    cudnnDataType_t dataType;
    cudnnTensorFormat_t tensorFormat;

    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc, biasTensorDesc;

    cudnnFilterDescriptor_t filterDesc;
    int Sx,Sy,Sz,Sd;
    int padH,padW;
    float alpha_conv = 1;
    float beta_conv = 0;
    float *filter_d,*bias_d;
    float alpha_add = 1;
    float beta_add = 1;
    cudnnConvolutionDescriptor_t convDesc;
};
}

#endif // GPU_CONVOLUTION_LAYER_H
