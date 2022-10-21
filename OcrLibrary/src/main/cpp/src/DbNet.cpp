#include "DbNet.h"
#include "OcrUtils.h"

DbNet::DbNet() {
#ifdef __VULKAN__
    net.opt.use_vulkan_compute = true;
    LOGI("DbNet use vulkan compute");
#endif
}

DbNet::~DbNet() {
    net.clear();
}

void DbNet::setNumThread(int numOfThread) {
    numThread = numOfThread;
}

bool DbNet::initModel(AAssetManager *mgr, const std::string &name) {
    int ret_param = net.load_param(mgr, (name + ".param").c_str());
    int ret_bin = net.load_model(mgr, (name + ".bin").c_str());
    if (ret_param != 0 || ret_bin != 0) {
        LOGE("# %d  %d", ret_param, ret_bin);
        return false;
    }
    return true;
}

std::vector<TextBox> findRsBoxes(const cv::Mat &predMat, const cv::Mat &dilateMat, ScaleParam &s,
                                 const float boxScoreThresh, const float unClipRatio) {
    const int longSideThresh = 3;//minBox 长边门限
    const int maxCandidates = 1000;

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(dilateMat, contours, hierarchy, cv::RETR_LIST,
                     cv::CHAIN_APPROX_SIMPLE);

    int numContours = contours.size() >= maxCandidates ? maxCandidates : contours.size();

    std::vector<TextBox> rsBoxes;

    for (int i = 0; i < numContours; i++) {
        if (contours[i].size() <= 2) {
            continue;
        }
        cv::RotatedRect minAreaRect = cv::minAreaRect(contours[i]);

        float longSide;
        std::vector<cv::Point2f> minBoxes = getMinBoxes(minAreaRect, longSide);

        if (longSide < longSideThresh) {
            continue;
        }

        float boxScore = boxScoreFast(minBoxes, predMat);
        if (boxScore < boxScoreThresh)
            continue;

        //-----unClip-----
        cv::RotatedRect clipRect = unClip(minBoxes, unClipRatio);
        if (clipRect.size.height < 1.001 && clipRect.size.width < 1.001) {
            continue;
        }
        //-----unClip-----

        std::vector<cv::Point2f> clipMinBoxes = getMinBoxes(clipRect, longSide);
        if (longSide < longSideThresh + 2)
            continue;

        std::vector<cv::Point> intClipMinBoxes;

        for (int p = 0; p < clipMinBoxes.size(); p++) {
            float x = clipMinBoxes[p].x / s.ratioWidth;
            float y = clipMinBoxes[p].y / s.ratioHeight;
            int ptX = (std::min)((std::max)(int(x), 0), s.srcWidth - 1);
            int ptY = (std::min)((std::max)(int(y), 0), s.srcHeight - 1);
            cv::Point point{ptX, ptY};
            intClipMinBoxes.push_back(point);
        }
        rsBoxes.push_back(TextBox{intClipMinBoxes, boxScore});
    }
    reverse(rsBoxes.begin(), rsBoxes.end());
    return rsBoxes;
}

std::vector<TextBox>
DbNet::getTextBoxes(cv::Mat &src, ScaleParam &s, float boxScoreThresh, float boxThresh, float unClipRatio) {
    cv::Mat srcResize;
    cv::resize(src, srcResize, cv::Size(s.dstWidth, s.dstHeight));
    ncnn::Mat input = ncnn::Mat::from_pixels(srcResize.data, ncnn::Mat::PIXEL_RGB,
                                             srcResize.cols, srcResize.rows);

    input.substract_mean_normalize(meanValues, normValues);
    ncnn::Extractor extractor = net.create_extractor();
    extractor.set_num_threads(numThread);
    extractor.input("input", input);
    ncnn::Mat out;
    extractor.extract("output", out);
    //-----Data preparation-----
    cv::Mat fMapMat(srcResize.rows, srcResize.cols, CV_32FC1);
    memcpy(fMapMat.data, (float *) out.data, srcResize.rows * srcResize.cols * sizeof(float));

    //-----boxThresh-----
    cv::Mat norfMapMat;
    norfMapMat = fMapMat > boxThresh;

    return findRsBoxes(fMapMat, norfMapMat, s, boxScoreThresh, unClipRatio);
}