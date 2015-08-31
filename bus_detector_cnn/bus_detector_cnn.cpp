#include "bus_detector_cnn.h"
#include "utility.h"


bus_detector_cnn::bus_detector_cnn(string modelFile)
{
    busDetector.load_model(modelFile,averageImage);
    cout<<"Load Setting Sucessfully"<<endl;
}
void bus_detector_cnn::findBus(string inputImagePath)
{

    Mat inputImage = imread(inputImagePath);
    //resize(inputImage,inputImage,Size(640,480));

    //Making a multiscale input image
    vector<float> processScale = {1.0,0.8,0.6,0.4,0.2};
    Mat scaledImage;
    vector<busBBox> allBusBoxes;
    for(uint scaleNo = 0 ; scaleNo < processScale.size() ; scaleNo++)
    {
        resize(inputImage,scaledImage,Size(),processScale[scaleNo],processScale[scaleNo]);

        //Suppress scale if images is smaller than CNN input size
        if(scaledImage.rows >= windowSizeH && scaledImage.cols >= windowSizeW)
        {
            vector<Rect> scaleBboxes;
            Mat rowWindowImage;                                 //sliding windows container
            slidingWindows(scaledImage,windowStrides,processScale[scaleNo],rowWindowImage,scaleBboxes);

            //#ifdef DEBUGF
            cout<<"Scale : "<<processScale[scaleNo]<<" Row Image Size: "<<rowWindowImage.rows<<" x "<<rowWindowImage.cols<<" Total windows:"<<scaleBboxes.size()<<endl;
            //#endif // DEBUGF
int64 t0 = cv::getTickCount();
            float *windows_h = (float*) rowWindowImage.data;    //pointer to sliding window in row-fashion format (NO-OP)
            float *windows_d;

            checkCudaErrors( cudaMalloc(&windows_d,scaleBboxes.size() * windowSizeH * windowSizeW * 3 *sizeof(float)));
            checkCudaErrors( cudaMemcpy(windows_d,windows_h,scaleBboxes.size() * 3 * windowSizeH * windowSizeW * sizeof(float),cudaMemcpyHostToDevice));

            uint nBatch = scaleBboxes.size() / concurrentWindows;

            float *output_d;
            float *output_h = new float[totalClass*scaleBboxes.size()];
            float *batchwindows_d;
            for(uint batch = 0 ; batch < nBatch ; batch++)
            {
                int n = concurrentWindows;                      //input data to cuDNN in this project is in NCHW format (Nimages-channels-row-col)
                int c = 3;
                int h = windowSizeH;
                int w = windowSizeW;

                batchwindows_d = &windows_d[batch * (concurrentWindows * c *h * w)];
                //Start Detector
                busDetector.feed_forward(&batchwindows_d,n,c,h,w,&output_d);

                //Copy back result (Be note that copy in block-format is faster than copy only desired elements)
                checkCudaErrors( cudaMemcpy(&output_h[batch * concurrentWindows * c],output_d,n * c * h * w * sizeof(float),cudaMemcpyDeviceToHost));
                checkCudaErrors( cudaFree(output_d));
            }

            int remainingWindows = scaleBboxes.size() - (concurrentWindows * nBatch);
            if (remainingWindows > 0)
            {
                int n = remainingWindows;                      //input data to cuDNN in this project is in NCHW format (Nimages-channels-row-col)
                int c = 3;
                int h = windowSizeH;
                int w = windowSizeW;

                batchwindows_d = &windows_d[nBatch * (concurrentWindows * c *h * w)];
                //Start Detector
                busDetector.feed_forward(&batchwindows_d,n,c,h,w,&output_d);

                //Copy back result (Be note that copy in block-format is faster than copy only desired elements)
                checkCudaErrors( cudaMemcpy(&output_h[nBatch * concurrentWindows * c],output_d,n * c * h * w * sizeof(float),cudaMemcpyDeviceToHost));
                checkCudaErrors( cudaFree(output_d));
            }
int64 t1 = cv::getTickCount();
double secs = (t1-t0)/cv::getTickFrequency();
            cout<<"GPU Time: "<<secs<<endl;
            checkCudaErrors( cudaFree(windows_d));
            pickBusBBoxes(scaleBboxes,output_h,allBusBoxes);
            delete [] output_h;
        }
        //inputImage too small
        else
        {
            break;
        }
    }

    //Non-correct version of nms (???)
    nms(allBusBoxes,0.4);


    //Display only high score box
    for(uint box = 0 ; box < allBusBoxes.size() ; box++)
    {
        if(allBusBoxes[box].keep && allBusBoxes[box].score > 0.6)
        {
            cout<<"["<<allBusBoxes[box].bbox.x<<" "<<allBusBoxes[box].bbox.y<<" "<<allBusBoxes[box].bbox.width<<" "<<allBusBoxes[box].bbox.height<<"] Score: "<<allBusBoxes[box].score<<endl;
            rectangle(inputImage,allBusBoxes[box].bbox,Scalar(255,255,255),1);
        }
    }
    imshow("Image",inputImage);
    imwrite("/home/thananop/1_out.jpg",inputImage);
    waitKey(0);
}
