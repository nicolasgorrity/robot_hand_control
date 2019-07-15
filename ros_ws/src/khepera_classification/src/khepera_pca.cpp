#include "khepera_pca.hpp"
#include "khepera_utils.hpp"

#include <fstream>
#include <ros/console.h>

#define PCA_NUMBER 25

void khepera::pca(std::vector<float>& X, std::vector<std::vector<float>>& pcaCoefficients, std::vector<float>& pcaMu, std::vector<float>& X_pca) {
    X_pca.resize(PCA_NUMBER);

    // PCA centers data
    for (unsigned int i=0; i<X.size(); ++i) {
        X[i] -= pcaMu[i];
    }

    // Multiply by inverse of PCA coefficients matrix
    for (unsigned int i=0; i<PCA_NUMBER; ++i) {
        float sum = 0;
        for (unsigned int j=0; j<pcaCoefficients[i].size(); ++j) {
            sum += X[j] * pcaCoefficients[j][i];
        }
        X_pca[i] = sum;
    }
}

void khepera::readPcaCoefficients(std::string filenameCoeff, std::vector<std::vector<float>>& pcaCoefficients, std::string filenameMu, std::vector<float>& pcaMu) {
    std::string line = "";
    std::string delimiter = ",";

    std::ifstream file(filenameCoeff);
    if (!file.is_open()) {
        ROS_ERROR("Could not open file %s", filenameCoeff.c_str());
        return;
    }

    while (getline(file, line)) {
        // Get the line of strings as a vector of floats
        std::vector<float> data;
        size_t sz = 0;
        size_t pos = 0;
        std::string token;
        while ((pos = line.find(delimiter)) != std::string::npos) {
            token = line.substr(0, pos);
            data.push_back(std::stof(token, &sz));
            line.erase(0, pos + delimiter.length());
        }
        data.push_back(std::stof(line, &sz));
		pcaCoefficients.push_back(data);
	}
    file.close();

    std::ifstream fileMu(filenameMu);
    if (!fileMu.is_open()) {
        ROS_ERROR("Could not open file %s", filenameMu.c_str());
        return;
    }

    while (getline(fileMu, line)) {
        // Get the line of strings as a vector of floats
        size_t sz = 0;
        size_t pos = 0;
        std::string token;
        while ((pos = line.find(delimiter)) != std::string::npos) {
            token = line.substr(0, pos);
            pcaMu.push_back(std::stof(token, &sz));
            line.erase(0, pos + delimiter.length());
        }
        pcaMu.push_back(std::stof(line, &sz));
	}
    fileMu.close();
}
