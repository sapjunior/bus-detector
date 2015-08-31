#ifndef GPU_CNN_H
#define GPU_CNN_H
#include <vector>
#include <memory>

#include "config_reader.h"
#include "gpu_layer.h"
#include <opencv2/opencv.hpp>

namespace gpu_cnn
{
class gpu_cnn_based
{
public:
    gpu_cnn_based() {}

    //Load Model From MAT File using matio
    void load_model(string modelFile,cv::Mat& averageImage)
    {
        checkCUDNN( cudnnCreate(&cudnnHandle));
        dataType = CUDNN_DATA_FLOAT;
        tensorFormat = CUDNN_TENSOR_NCHW;

        config_reader cnn_config;
        if(cnn_config.read(modelFile))
        {
            //Preprocess Layer (subtract averageImage)
            float *averageImage_d;
            size_t averageImage_elems_size = 224*224*3*sizeof(float);
            checkCudaErrors( cudaMalloc(&averageImage_d,averageImage_elems_size));
            checkCudaErrors( cudaMemcpy(averageImage_d,cnn_config.averageImage,averageImage_elems_size,cudaMemcpyHostToDevice));
            gpu_preprocess_layer Pre_Layer(cudnnHandle,dataType,tensorFormat,averageImage_d);
            add(Pre_Layer);

            for(int nlayers = 0 ; nlayers< cnn_config.nlayers; nlayers++)
            {
                if(cnn_config.filterLayers[nlayers].type == "conv")
                {
                    float *filter_d,*bias_d;
                    size_t filter_elems_size = cnn_config.filterLayers[nlayers].Sx*cnn_config.filterLayers[nlayers].Sy*cnn_config.filterLayers[nlayers].Sz*cnn_config.filterLayers[nlayers].Sd*sizeof(float);
                    checkCudaErrors( cudaMalloc(&filter_d,filter_elems_size));
                    checkCudaErrors( cudaMemcpy(filter_d,cnn_config.filterLayers[nlayers].filters,filter_elems_size,cudaMemcpyHostToDevice));

                    size_t bias_elems_size = cnn_config.filterLayers[nlayers].Sd*sizeof(float);
                    checkCudaErrors( cudaMalloc(&bias_d,bias_elems_size));
                    checkCudaErrors( cudaMemcpy(bias_d,cnn_config.filterLayers[nlayers].biases,bias_elems_size,cudaMemcpyHostToDevice));

                    gpu_convolution_layer C_Layer(cudnnHandle,dataType,tensorFormat,
                                                  cnn_config.filterLayers[nlayers].Sz,cnn_config.filterLayers[nlayers].Sd,cnn_config.filterLayers[nlayers].Sx,cnn_config.filterLayers[nlayers].Sy,cnn_config.filterLayers[nlayers].pad,
                                                  filter_d,bias_d);
                    add(C_Layer);
                }
                else if(cnn_config.filterLayers[nlayers].type == "pool")
                {
                    gpu_cnn::gpu_pooling_layer P_Layer(cudnnHandle,dataType,tensorFormat,
                                                       cnn_config.filterLayers[nlayers].poolSize,cnn_config.filterLayers[nlayers].poolSize,
                                                       cnn_config.filterLayers[nlayers].poolSize,cnn_config.filterLayers[nlayers].poolSize);
                    add(P_Layer);
                }
                else if(cnn_config.filterLayers[nlayers].type == "relu")
                {
                    gpu_cnn::gpu_relu_activatefunc R_Layer(cudnnHandle,dataType,tensorFormat);
                    add(R_Layer);
                }
                else if(cnn_config.filterLayers[nlayers].type == "softmax")
                {
                    gpu_cnn::gpu_softmax_activatefunc S_Layer(cudnnHandle,dataType,tensorFormat);
                    add(S_Layer);
                }
            }

            cnn_config.release();
        }
        else
        {
            cout<<"Load Config Failed"<<endl;
            exit(-1);
        }
    }
    void feed_forward(float** input_d,int& n,int& c,int& w,int& h,float** output_d)
    {
        //inputImage_d as devicePointer , n = number of inputImage , w-h size of inputImage
        for(unsigned int currentLayer = 0 ; currentLayer < layers.size() ; currentLayer++)
        {
            (*layers[currentLayer]).feed_forward(&input_d,n,c,h,w,&output_d);
            if(currentLayer>1)
                checkCudaErrors( cudaFree(*input_d));
            //Swap input_d and output_d
            if(currentLayer+1 < layers.size())
            {
                *input_d = *output_d;
                *output_d = *input_d;
            }
        }
    }

    virtual ~gpu_cnn_based()
    {
        //Cleaning Up
        for(unsigned int currentLayer = 0 ; currentLayer < layers.size() ; currentLayer++)
            (*layers[currentLayer]).cleanup();
        checkCUDNN( cudnnDestroy(cudnnHandle));
        cout<<"Cleaning Up"<<endl;
    }
private:
    cudnnHandle_t cudnnHandle;
    cudnnDataType_t dataType;
    cudnnTensorFormat_t tensorFormat;
    vector <shared_ptr<gpu_based_layer> > layers;
    float **tempPtr_d;
    template <typename T>
    void add(T _layer)
    {
        (layers).push_back(make_shared<T>(_layer));
    }
};
}
#endif // GPU_CNN_H
