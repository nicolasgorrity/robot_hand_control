#include "khepera_fourier_transform.hpp"
#include "khepera_utils.hpp"

#include <cmath>

void khepera::fourierDescriptors(std::vector<std::complex<float>>& contour, std::vector<std::complex<float>>& descriptors) {
    // Compute mean of pixels
    std::complex<float> mean = utils::mean(contour);

    // Substract mean to elements of vector
    utils::substract(contour, mean);

    // Compute whole FFT
    std::vector<std::complex<float>> fourierTransform;
    cv::dft(contour, fourierTransform, cv::DFT_SCALE + cv::DFT_COMPLEX_OUTPUT);

    // Resize descriptors vector
    descriptors.resize(2*cmax + 1);

    // Fill it with relevant elements of Fourier Transform
    unsigned int lenFourier = fourierTransform.size();
    for (unsigned int i=0; i<cmax+1; ++i) {
        descriptors[i] = fourierTransform[cmax-i];
    }
    for (unsigned int i=0; i<cmax; ++i) {
        descriptors[cmax+1+i] = fourierTransform[lenFourier-1-i];
    }

    // Inverse sequence order if the browse is done in non-trignonometric direction
    if (std::abs(descriptors[cmax-1]) > std::abs(descriptors[cmax+1])) {
        std::reverse(descriptors.begin(), descriptors.end());
    }

    // Correct phase to normalize wrt rotation
    float phi = std::arg(descriptors[cmax-1] * descriptors[cmax+1]) / 2.0;
    std::complex<float> i_(0, 1);
    utils::multiply(descriptors, std::exp(-phi*i_));
    float theta = std::arg(descriptors[cmax+1]);
    for (int k=0; k<descriptors.size(); ++k) {
        descriptors[k] *= std::exp(-i_*(float)((k-cmax)*theta));
    }

    // Correct norm to normalize wrt homothecy
    float factor = std::abs(descriptors[cmax+1]);
    for (auto it=descriptors.begin(); it!=descriptors.end(); ++it) {
        (*it) /= factor;
    }
}

void khepera::descriptorsToFeatures(const std::vector<std::complex<float>>& descriptors, std::vector<float>& X) {
    unsigned int nbDescriptors = descriptors.size();
    X.resize(2 * nbDescriptors);
    for (unsigned int i=0; i<nbDescriptors; ++i) {
        X[i] = descriptors[i].real();
        X[nbDescriptors + i] = descriptors[i].imag();
    }
}
