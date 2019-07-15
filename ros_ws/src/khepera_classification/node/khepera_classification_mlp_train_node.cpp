#include "ros/ros.h"
#include <ros/package.h>
#include <std_msgs/String.h>
#include <opencv2/opencv.hpp>

#include "khepera_classification.hpp"
#include "khepera_classif_mlp.hpp"

int main(int argc, char **argv) {
    ros::init(argc, argv, "khepera_classification");
    std::string packagePath = ros::package::getPath("khepera_classification");
    ros::NodeHandle n;

    // Load train dataset (only need to be done once)
    Mat_<float> Xtrain, Ytrain;
    Mat_<float> Xval, Yval;
    float train_val_split = 0.8;
    khepera::loadMlpTrainingSet(packagePath + X_TRAIN_FILE_PATH, Xtrain, Xval,
                                packagePath + Y_TRAIN_FILE_PATH, Ytrain, Yval,
                                train_val_split);

    // Fit classifier
    Ptr<ANN_MLP> network;
    ros::WallTime start = ros::WallTime::now();
    bool success = khepera::trainMultiLayerPerceptron(network, Xtrain, Ytrain);
    ros::WallTime end = ros::WallTime::now();
    if (!success) {
        ROS_ERROR("Neural network failed during training step which returned false");
        return -1;
    }

    // Test classifier
    float val_loss = khepera::testMultiLayerPerceptron(network, Xval, Yval);

    // Save classifier
    khepera::saveMultiLayerPerceptron(packagePath + MLP_FILE_PATH, network);

    double training_time = (end - start).toNSec() * 1e-6;
    ROS_INFO("Multi-Layer Perceptron trained in %.0f ms", training_time);
    ROS_INFO("Model saved in %s", std::string(packagePath + MLP_FILE_PATH).c_str());
    ROS_INFO("Validation loss is %.6f %% for a train/validation split of %d%%/%d%%",
             val_loss*100, (int)(train_val_split*100), 100-(int)(train_val_split*100));

    return 0;
}
