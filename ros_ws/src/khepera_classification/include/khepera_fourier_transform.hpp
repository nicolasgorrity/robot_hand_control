#pragma once
#include <opencv2/opencv.hpp>

#define cmax 10

namespace khepera {

    void fourierDescriptors(std::vector<std::complex<float>>& contour, std::vector<std::complex<float>>& descriptors);
    void descriptorsToFeatures(const std::vector<std::complex<float>>& descriptors, std::vector<float>& X);

}
