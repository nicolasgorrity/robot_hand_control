#include <ros/console.h>
#include "khepera_camera.hpp"

void khepera::getCameraImage(cv::VideoCapture& cap, cv::Mat& out) {
    cap >> out;
    cv::cvtColor(out, out, cv::COLOR_BGR2RGB);
}

bool khepera::initCamera(cv::VideoCapture& cap) {
    if (!cap.isOpened()) {
        ROS_ERROR("khepera_camera.cpp::initCamera() --> Could not open camera flux");
        return false;
    }
    return true;
}
