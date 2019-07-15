#include "khepera_classification.hpp"
#include <fstream>
#include <ros/console.h>

void khepera::readContourMsg(const khepera_classification::Contour::ConstPtr& contourMsg, std::vector<std::complex<float>>& contour) {
    for (int i=0; i<contourMsg->contour.size(); ++i) {
        const khepera_classification::Pixel& pixel = contourMsg->contour[i];
        contour.push_back(std::complex<float>(pixel.x, pixel.y));
    }
}
