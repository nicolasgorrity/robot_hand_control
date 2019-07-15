#include "khepera_classif_mlp.hpp"
#include <fstream>
#include <ros/console.h>

bool khepera::trainMultiLayerPerceptron(Ptr<ANN_MLP>& network, Mat_<float>& Xtrain, Mat_<float>& Ytrain) {
    // Create the neural network
    network = ANN_MLP::create();

    // Configure sizes of input, hidden and output layers
    Mat_<int> layerSizes(1, 3);
    layerSizes(0, 0) = Xtrain.cols;     // Input
    layerSizes(0, 1) = 10;              // Hidden
    layerSizes(0, 2) = Ytrain.cols;     // Output

    // Set network architecture and parameters
    network->setLayerSizes(layerSizes);
    network->setActivationFunction(ANN_MLP::SIGMOID_SYM);
    network->setTrainMethod(ANN_MLP::BACKPROP, 0.01);
    TermCriteria criteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, 0.000000001);
    network->setTermCriteria(criteria);
    Ptr<TrainData> trainData = TrainData::create(Xtrain, ROW_SAMPLE, Ytrain);

    // Train the network
    return network->train(trainData);
}

float khepera::testMultiLayerPerceptron(Ptr<ANN_MLP>& network, Mat_<float>& Xval, Mat_<float>& Yval) {
    unsigned int nb_predictions = 0;
    unsigned int nb_true_pred = 0;
    if (Xval.rows == 0) return -1;

    for (int i=0; i<Xval.rows; ++i) {
        std::vector<float> X;
        for (int j=0; j<Xval.cols; ++j) X.push_back(Xval[i][j]);
        Mat result;
        khepera::predictWithMLP(network, X, result);
        unsigned short predictedLabel = khepera::retrieveMlpPrediction(result);
        ++nb_predictions;
        if (Yval[i][predictedLabel-1] == 1.0) ++nb_true_pred;
    }

    return 1.0 - float(nb_true_pred) / float(nb_predictions);
}

void khepera::saveMultiLayerPerceptron(std::string filename, Ptr<ANN_MLP>& network) {
    network->save(filename);
}

void khepera::loadMultiLayerPerceptron(std::string filename, Ptr<ANN_MLP>& network) {
    network = Algorithm::load<ANN_MLP>(filename);
}

void khepera::predictWithMLP(Ptr<ANN_MLP>& network, std::vector<float>& X, Mat& result) {
    Mat_<float> Xpredict(1, X.size());
    for (int i=0; i<X.size(); i++) {
        Xpredict.at<float>(0, i) = X[i];
    }
    network->predict(Xpredict, result);
}

unsigned short khepera::retrieveMlpPrediction(Mat& result) {
    unsigned int max_idx = 0;
    float max_value = 0;
    for (int i=0; i<result.cols; ++i) {
        float val = result.at<float>(0, i);
        if (val >= max_value) {
            max_idx = i;
            max_value = val;
        }
    }
    return max_idx + 1;
}

void khepera::loadMlpTrainingSet(std::string filenameX, Mat_<float>& Xtrain, Mat_<float>& Xval,
                                 std::string filenameY, Mat_<float>& Ytrain, Mat_<float>& Yval,
                                 float train_val_split) {
    std::string line = "";
    std::string delimiter = ",";

    std::vector<std::vector<float>> X;
    std::ifstream file(filenameX);
    if (!file.is_open()) {
        std::cout << "Could not open file " << filenameX << std::endl;
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
        X.push_back(data);
    }
    file.close();

    std::vector<unsigned char> y;
    std::ifstream fileY(filenameY);
    if (!fileY.is_open()) {
        std::cout << "Could not open file " << filenameY << std::endl;
        return;
    }

    while (getline(fileY, line)) {
        // Get the line of strings as a vector of floats
        size_t sz = 0;
        size_t pos = 0;
        std::string token;
        while ((pos = line.find(delimiter)) != std::string::npos) {
            token = line.substr(0, pos);
            y.push_back(std::stoi(token, &sz));
            line.erase(0, pos + delimiter.length());
        }
        y.push_back(std::stof(line, &sz));
    }
    fileY.close();

    std::vector<std::vector<float>> Y;
    for (unsigned int i=0; i<y.size(); ++i) Y.push_back(std::vector<float>(NB_CLUSTER, 0.0));
    for (unsigned int i=0; i<y.size(); ++i) Y[i][y[i]-1] = 1.0;

    int nb_train_samples = (int)(X.size() * train_val_split);
    int nb_val_samples = X.size() - nb_train_samples;
    Xtrain = Mat_<float>(nb_train_samples, X[0].size());
    Ytrain = Mat_<float>(nb_train_samples, Y[0].size());
    Xval   = Mat_<float>(nb_val_samples  , X[0].size());
    Yval   = Mat_<float>(nb_val_samples  , Y[0].size());
    for (int i=0; i<nb_train_samples; ++i) {
        for (int j=0; j<Xtrain.cols; ++j) {
            Xtrain.at<float>(i,j) = X[i][j];
        }
    }
    for (int i=0; i<nb_val_samples; ++i) {
        for (int j=0; j<Xval.cols; ++j) {
            Xval.at<float>(i,j) = X[nb_train_samples+i][j];
        }
    }

    for (int i=0; i<nb_train_samples; ++i) {
        for (int j=0; j<Ytrain.cols; ++j) {
            Ytrain.at<float>(i,j) = Y[i][j];
        }
    }
    for (int i=0; i<nb_val_samples; ++i) {
        for (int j=0; j<Yval.cols; ++j) {
            Yval.at<float>(i,j) = Y[nb_train_samples+i][j];
        }
    }
}
