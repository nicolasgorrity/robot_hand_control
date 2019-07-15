#pragma once

#include <vector>
#include <string>

namespace khepera {

    void pca(std::vector<float>& X, std::vector<std::vector<float>>& pcaCoefficients, std::vector<float>& pcaMu, std::vector<float>& X_pca);
    void readPcaCoefficients(std::string filenameCoeff, std::vector<std::vector<float>>& pcaCoefficients, std::string filenameMu, std::vector<float>& pcaMu);

}
