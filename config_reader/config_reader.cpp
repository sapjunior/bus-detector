#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <matio.h>

#include "config_reader.h"

bool config_reader::read(string config_filepath)
{
    mat_t *config_filePtr;


    config_filePtr = Mat_Open(config_filepath.c_str(),MAT_ACC_RDONLY);
    //Pointer to layer
    matvar_t *mlayers;
    mlayers = Mat_VarRead(config_filePtr,"layers");

    //Numbers of Layers
    nlayers = mlayers->dims[1];
    filterLayers = new _layers[nlayers];
    /*===========Run through each layers==========*/
    matvar_t *layer_elementPtr;              //Layer Pointer

    for (int cIdx = 0; cIdx < nlayers; cIdx++)
    {
        layer_elementPtr = Mat_VarGetCell(mlayers,cIdx);

        matvar_t *layerType = Mat_VarGetStructField(layer_elementPtr,(char*)"type",MAT_BY_NAME,0);
        filterLayers[cIdx].type.append((char*)layerType->data,layerType->dims[1]);
        Mat_VarFree(layerType);
        //Convolution Layers
        if(filterLayers[cIdx].type=="conv")
        {

            //Read filters weights
            matvar_t *mFilters = Mat_VarGetStructField(layer_elementPtr,(char*)"filters",MAT_BY_NAME,0);
            float *Filters = (float*)mFilters->data;
            int nFiltersElems = mFilters->dims[0]*mFilters->dims[1]*mFilters->dims[2]*mFilters->dims[3];
            filterLayers[cIdx].Sx = mFilters->dims[0];
            filterLayers[cIdx].Sy = mFilters->dims[1];
            filterLayers[cIdx].Sz = mFilters->dims[2];
            filterLayers[cIdx].Sd = mFilters->dims[3];
            //Implement It as 1D Array
            filterLayers[cIdx].filters = new float[nFiltersElems];
            for(int nElems=0; nElems<nFiltersElems; nElems++)
            {
                filterLayers[cIdx].filters[nElems] = *Filters++;
            }
            Mat_VarFree(mFilters);

            //Read filter biases
            matvar_t *mBiases = Mat_VarGetStructField(layer_elementPtr,(char*)"biases",MAT_BY_NAME,0);
            float *Biases = (float*)mBiases->data;
            int nBiasesElems = mBiases->dims[1];
            filterLayers[cIdx].biases = new float[nBiasesElems];
            for(int nElems=0; nElems<nBiasesElems; nElems++)
            {
                filterLayers[cIdx].biases[nElems] = *Biases++;
            }
            Mat_VarFree(mBiases);

            matvar_t *mpadSize = Mat_VarGetStructField(layer_elementPtr,(char*)"pad",MAT_BY_NAME,0);
            double *padSize = (double*)mpadSize->data;
            filterLayers[cIdx].pad = *padSize;
            Mat_VarFree(mpadSize);

        }
        else if(filterLayers[cIdx].type == "pool")
        {
            matvar_t *mpoolType = Mat_VarGetStructField(layer_elementPtr,(char*)"method",MAT_BY_NAME,0);
            filterLayers[cIdx].poolType.append((char*)mpoolType->data);
            Mat_VarFree(mpoolType);

            matvar_t *mpoolSize = Mat_VarGetStructField(layer_elementPtr,(char*)"pool",MAT_BY_NAME,0);
            double *poolSize = (double*)mpoolSize->data;
            filterLayers[cIdx].poolSize = *poolSize;
            Mat_VarFree(mpoolSize);

            matvar_t *mpadSize = Mat_VarGetStructField(layer_elementPtr,(char*)"pad",MAT_BY_NAME,0);
            double *padSize = (double*)mpadSize->data;
            filterLayers[cIdx].pad = *padSize;
            Mat_VarFree(mpadSize);
        }
    }
    matvar_t *mNormalization = Mat_VarRead(config_filePtr,"normalization");
    matvar_t *maverageImage = Mat_VarGetStructField(mNormalization,(char*)"averageImage",MAT_BY_NAME,0);
    float *PaverageImage = (float*)maverageImage->data;
    int naverageImageElems = maverageImage->dims[0]*maverageImage->dims[1]*maverageImage->dims[2];
    averageImage = new float[naverageImageElems];
    for(int nElems=0; nElems<naverageImageElems; nElems++)
    {
        averageImage[nElems] = *PaverageImage++;
    }
    Mat_VarFree(maverageImage);

    //Don't know why can't free these var. May be matio bug?
    //Mat_VarFree(mNormalization);
    //Mat_VarFree(mlayers);

    Mat_Close(config_filePtr);
    return true;
}

void config_reader::release()
{

    for (int nLayer = 0; nLayer < nlayers; nLayer++)
    {
        if(filterLayers[nLayer].type=="conv")
        {
            delete[] filterLayers[nLayer].biases;
            delete[] filterLayers[nLayer].filters;
        }
    }
    delete[] filterLayers;
    delete[] averageImage;
}
