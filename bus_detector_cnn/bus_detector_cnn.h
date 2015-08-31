#ifndef BUS_DETECTOR_CNN_H
#define BUS_DETECTOR_CNN_H
#define windowSizeH 224
#define windowSizeW 224
#define windowStrides 30
#define totalClass 1000
#define concurrentWindows 32

#include <opencv2/opencv.hpp>
#include "gpu_cnn.h"
using namespace cv;
class bus_detector_cnn
{
    public:
        bus_detector_cnn(string modelFile);
        void findBus(string inputImagePath);

    protected:
    private:
        gpu_cnn::gpu_cnn_based busDetector;
        Mat averageImage;



};
struct busBBox
        {
            Rect bbox;
            float score;
            int area;
            bool keep = true;
        };
#endif // BUS_DETECTOR_CNN_H
