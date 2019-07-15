#pragma once
#include <opencv2/opencv.hpp>

namespace khepera {

    void getCameraImage(cv::VideoCapture& cap, cv::Mat& out);
    bool initCamera(cv::VideoCapture& cap);

}
