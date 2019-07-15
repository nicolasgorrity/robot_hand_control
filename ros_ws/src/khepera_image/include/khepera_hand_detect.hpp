#pragma once
#include <opencv2/opencv.hpp>

namespace khepera {

    void extractHand(const cv::Mat& in, cv::Mat& out, unsigned char& seuil, cv::BackgroundSubtractor *pMOG2, bool& backgroundRemove);

}
