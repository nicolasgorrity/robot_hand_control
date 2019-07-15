#include "khepera_classif_knn.hpp"
#include <fstream>
#include <ros/console.h>

void khepera::loadKnnTrainingSet(std::string filenameX, knn::Matrix<float>& Xtrain, std::string filenameY, knn::Vector<unsigned short>& Ytrain) {
    std::string line = "";
    std::string delimiter = ",";

    std::ifstream file(filenameX);
    if (!file.is_open()) {
        ROS_ERROR("Could not open file %s", filenameX.c_str());
        return;
    }

    while (getline(file, line)) {
        // Get the line of strings as a vector of floats
        knn::Vector<float> data;
        size_t sz = 0;
        size_t pos = 0;
        std::string token;
        while ((pos = line.find(delimiter)) != std::string::npos) {
            token = line.substr(0, pos);
            data.push_back(std::stof(token, &sz));
            line.erase(0, pos + delimiter.length());
        }
        data.push_back(std::stof(line, &sz));
        Xtrain.push_back(data);
    }
    file.close();

    std::ifstream fileY(filenameY);
    if (!fileY.is_open()) {
        ROS_ERROR("Could not open file %s", filenameY.c_str());
        return;
    }

    while (getline(fileY, line)) {
        // Get the line of strings as a vector of floats
        size_t sz = 0;
        size_t pos = 0;
        std::string token;
        while ((pos = line.find(delimiter)) != std::string::npos) {
            token = line.substr(0, pos);
            Ytrain.push_back(std::stoi(token, &sz));
            line.erase(0, pos + delimiter.length());
        }
        Ytrain.push_back(std::stof(line, &sz));
    }
    fileY.close();
}
