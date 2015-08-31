#include <opencv2/opencv.hpp>
#include "bus_detector_cnn.h"

using namespace cv;
void slidingWindows(Mat scaledImage,int strides,float scale,Mat& splitScaledImage,vector<Rect>& bbox){

    vector<Mat> bgrChannel;

    scaledImage.convertTo(scaledImage,CV_32F);

    split(scaledImage,bgrChannel);

    //Non-Optimize Version [To be correct later cause I AM VERY SLEEPY NOW]
    Rect currentRoi,originalscaleRoi;
    for(int row = 0 ; row < scaledImage.rows-windowSizeH+1 ; row = row+strides)
    {
        for(int col = 0;col < scaledImage.cols-windowSizeW+1 ; col = col+strides)
        {
            currentRoi = Rect(col,row,windowSizeW,windowSizeH);
            splitScaledImage.push_back(bgrChannel[2](currentRoi));
            splitScaledImage.push_back(bgrChannel[1](currentRoi));
            splitScaledImage.push_back(bgrChannel[0](currentRoi));

            //Not so good scaling,just an approximation for faster implementation
            originalscaleRoi.x = currentRoi.x/scale;
            originalscaleRoi.y = currentRoi.y/scale;
            originalscaleRoi.width = currentRoi.width/scale;
            originalscaleRoi.height = currentRoi.height/scale;
            bbox.push_back(originalscaleRoi);
        }
    }
}

void pickBusBBoxes(vector<Rect>& scaleBusBBoxes,float*& score,vector<busBBox>& allBusBBoxes){
    for(uint box = 0; box < scaleBusBBoxes.size() ; box++)
    {
        //Roughly approximation prob "minibus" & "trolleybus"
        if(max(score[box*totalClass+654],score[box*totalClass+874]) > 0.5)
        {
            busBBox bus;
            bus.bbox = scaleBusBBoxes[box];
            bus.area = scaleBusBBoxes[box].area();
            bus.score = max(score[box*totalClass+654],score[box*totalClass+874]);
            allBusBBoxes.push_back(bus);
        }
    }

}

//Comapred funtion (STL-Style)
bool compareByScore(const busBBox &a, const busBBox &b)
{
    return a.score > b.score;
}
//NMS
void nms(vector<busBBox>& allBusBBoxes,float overlap){
    std::sort(allBusBBoxes.begin(), allBusBBoxes.end(), compareByScore);
    for(uint box = 0 ; box < allBusBBoxes.size() ; box++)
    {
        if(allBusBBoxes[box].keep){
            for(uint compareBox = box + 1 ; compareBox < allBusBBoxes.size() ; compareBox++)
            {
                 if(allBusBBoxes[compareBox].keep){
                    //Compute overlap ratio
                    float overlapRatio = (float)(allBusBBoxes[box].bbox & allBusBBoxes[compareBox].bbox).area() / (float)(allBusBBoxes[compareBox].bbox.area());
                    if(overlapRatio > overlap)
                        allBusBBoxes[compareBox].keep = false;
                 }
            }
        }
    }
}
