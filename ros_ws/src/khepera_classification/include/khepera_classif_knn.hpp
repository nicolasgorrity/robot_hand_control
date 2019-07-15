#pragma once

#include "khepera_knn.hpp"

namespace khepera {

    void loadKnnTrainingSet(std::string filenameX, knn::Matrix<float>& Xtrain, std::string filenameY, knn::Vector<unsigned short>& Ytrain);

}
