#ifndef __OCR_CRNNNET_H__
#define __OCR_CRNNNET_H__

#include "OcrStruct.h"
#include <net.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

class CrnnNet {
public:

    ~CrnnNet();

    void setNumThread(int numOfThread);

    bool initModel(AAssetManager *mgr, const std::string &name, const std::string &keysName);

    std::vector<TextLine> getTextLines(std::vector<cv::Mat> &partImg);

private:
    int numThread;
    ncnn::Net net;

    const float meanValues[3] = {127.5, 127.5, 127.5};
    const float normValues[3] = {1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5};
    const int dstHeight = 48;

    std::vector<std::string> keys;

    TextLine scoreToTextLine(const std::vector<float> &outputData, int h, int w);

    TextLine getTextLine(cv::Mat &src);
};


#endif //__OCR_CRNNNET_H__
