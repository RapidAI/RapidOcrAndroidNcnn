#include "AngleNet.h"
#include "OcrUtils.h"
#include <numeric>

AngleNet::~AngleNet() {
    net.clear();
}

void AngleNet::setNumThread(int numOfThread) {
    numThread = numOfThread;
}

bool AngleNet::initModel(AAssetManager *mgr, const std::string &name) {
    int ret_param = net.load_param(mgr, (name + ".param").c_str());
    int ret_bin = net.load_model(mgr, (name + ".bin").c_str());
    if (ret_param != 0 || ret_bin != 0) {
        LOGE("# %d  %d", ret_param, ret_bin);
        return false;
    }
    return true;
}

Angle scoreToAngle(const std::vector<float> &outputData) {
    int maxIndex = 0;
    float maxScore = 0;
    for (int i = 0; i < outputData.size(); i++) {
        if (outputData[i] > maxScore) {
            maxScore = outputData[i];
            maxIndex = i;
        }
    }
    return {maxIndex, maxScore};
}

Angle AngleNet::getAngle(cv::Mat &src) {
    ncnn::Mat input = ncnn::Mat::from_pixels(
            src.data, ncnn::Mat::PIXEL_RGB,
            src.cols, src.rows);
    input.substract_mean_normalize(meanValues, normValues);
    ncnn::Extractor extractor = net.create_extractor();
    extractor.set_num_threads(numThread);
    extractor.input("input", input);
    ncnn::Mat out;
    extractor.extract("output", out);
    float *floatArray = (float *) out.data;
    std::vector<float> outputData(floatArray, floatArray + out.w);
    return scoreToAngle(outputData);
}

std::vector<Angle>
AngleNet::getAngles(std::vector<cv::Mat> &partImgs, bool doAngle, bool mostAngle) {
    auto size = partImgs.size();
    std::vector<Angle> angles(size);
    if (doAngle) {
        for (int i = 0; i < size; ++i) {
            double startAngle = getCurrentTime();
            cv::Mat angleImg;
            cv::resize(partImgs[i], angleImg, cv::Size(dstWidth, dstHeight));
            Angle angle = getAngle(angleImg);
            double endAngle = getCurrentTime();
            angle.time = endAngle - startAngle;
            angles[i] = angle;
        }
    } else {
        for (int i = 0; i < size; ++i) {
            angles[i] = Angle{-1, 0.f};
        }
    }
    //Most Possible AngleIndex
    if (doAngle && mostAngle) {
        auto angleIndexes = getAngleIndexes(angles);
        double sum = std::accumulate(angleIndexes.begin(), angleIndexes.end(), 0.0);
        double halfPercent = angles.size() / 2.0f;
        int mostAngleIndex;
        if (sum < halfPercent) {//all angle set to 0
            mostAngleIndex = 0;
        } else {//all angle set to 1
            mostAngleIndex = 1;
        }
        //printf("Set All Angle to mostAngleIndex(%d)\n", mostAngleIndex);
        for (int i = 0; i < angles.size(); ++i) {
            Angle angle = angles[i];
            angle.index = mostAngleIndex;
            angles.at(i) = angle;
        }
    }

    return angles;
}