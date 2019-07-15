#pragma once
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>

#define MLP_FILE_PATH "/files/neuralNetwork.xml"
#define NB_CLUSTER 4

using namespace cv;
using namespace cv::ml;

namespace khepera {

    void loadMlpTrainingSet(std::string filenameX, Mat_<float>& Xtrain, Mat_<float>& Xval, std::string filenameY, Mat_<float>& Ytrain, Mat_<float>& Yval, float train_val_split);
    bool trainMultiLayerPerceptron(Ptr<ANN_MLP>& network, Mat_<float>& Xtrain, Mat_<float>& Ytrain);
    float testMultiLayerPerceptron(Ptr<ANN_MLP>& network, Mat_<float>& Xval, Mat_<float>& Yval);
    void saveMultiLayerPerceptron(std::string filename, Ptr<ANN_MLP>& network);
    void loadMultiLayerPerceptron(std::string filename, Ptr<ANN_MLP>& network);
    void predictWithMLP(Ptr<ANN_MLP>& network, std::vector<float>& X, Mat& result);
    unsigned short retrieveMlpPrediction(Mat& result);

}
