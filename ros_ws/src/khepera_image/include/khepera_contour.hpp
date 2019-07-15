#pragma once
#include <opencv2/opencv.hpp>

namespace khepera {

    void extractContour(const cv::Mat& in, std::vector<cv::Point>& contour);
    void findContourOrientation(std::vector<cv::Point>& contour, std::string& orientation);
    void drawContour(cv::Mat& out, std::vector<cv::Point>& contour);

}
