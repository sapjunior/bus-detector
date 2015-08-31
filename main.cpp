#include <iostream>
#include <opencv2/opencv.hpp>
#include "bus_detector_cnn.h"
using namespace std;
using namespace cv;
int main()
{
    bus_detector_cnn n("/home/thananop/Works/Pre-Doctoral/matconvnet_converter/imagenet-vgg-verydeep-16_cudnn.mat");
    n.findBus("1.jpg");
    return 0;
}
