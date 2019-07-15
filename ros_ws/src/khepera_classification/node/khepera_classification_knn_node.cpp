#include "ros/ros.h"
#include <ros/package.h>
#include <std_msgs/String.h>
#include <opencv2/opencv.hpp>

#include "khepera_classification.hpp"
#include "khepera_fourier_transform.hpp"
#include "khepera_pca.hpp"
#include "khepera_classif_knn.hpp"

void callback(const boost::shared_ptr<khepera_classification::Contour const> msg,
              std::vector<std::vector<float>>& pcaCoefficients,
              std::vector<float>& pcaMu,
              knn::KnnClassifier<float, unsigned short>& clf,
              unsigned int& k,
              ros::Publisher& commands_publisher) {
    // Retrieve contour from message
    std::vector<std::complex<float>> contour;
    khepera::readContourMsg(msg, contour);

    if (contour.size() > 2*cmax+1) {
        // Get Fourier Descriptors of contour
        std::vector<std::complex<float>> descriptors;
        khepera::fourierDescriptors(contour, descriptors);

        // Separate real and imaginary parts of descriptors
        std::vector<float> X;
        khepera::descriptorsToFeatures(descriptors, X);

        // Transform with PCA
        std::vector<float> X_pca;
        khepera::pca(X, pcaCoefficients, pcaMu, X_pca);

        // Predict with KNN classifier
        knn::Matrix<float> Xpredict;
        Xpredict.push_back(X_pca);
        clf.predict(Xpredict, k);
        unsigned short predictedLabel = clf.getPrediction()[0];

        // Send message
        std::string command;
        switch (predictedLabel) {
        case FORWARD_LABEL:
            command = FORWARD;
            break;
        case TURN_LABEL:
            command = msg->orientation;
            break;
        case STOP_LABEL:
        case TRASH_LABEL:
        default:
            command = NONE;
        }
        std_msgs::String message;
        message.data = command;
        commands_publisher.publish(message);
        ROS_INFO("Classified command as %s", command.c_str());
    }
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "khepera_classification");
    std::string packagePath = ros::package::getPath("khepera_classification");
    ros::NodeHandle n;

    // Publisher of string commands
    ros::Publisher commands_publisher = n.advertise<std_msgs::String>("/khepera/string_commands", 1000);

    // Read PCA coefficients (only needs to be done once)
    std::vector<std::vector<float>> pcaCoefficients;
    std::vector<float> pcaMu;
    khepera::readPcaCoefficients(packagePath + COEFF_FILE_PATH, pcaCoefficients, packagePath + MU_FILE_PATH, pcaMu);

    // Load train dataset (only need to be done once)
    knn::Matrix<float> Xtrain;
    knn::Vector<unsigned short> Ytrain;
    khepera::loadKnnTrainingSet(packagePath + X_TRAIN_FILE_PATH, Xtrain, packagePath + Y_TRAIN_FILE_PATH, Ytrain);

    // Fit classifier
    knn::KnnClassifier<float, unsigned short> clf;
    unsigned int k = 6;
    clf.fit(Xtrain, Ytrain);

    // Subscriber of contour messages
    ros::Subscriber sub = n.subscribe<khepera_classification::Contour>("khepera/contours", 1000,
                                boost::bind(callback, _1,
                                            boost::ref(pcaCoefficients), boost::ref(pcaMu),
                                            boost::ref(clf), boost::ref(k),
                                            boost::ref(commands_publisher)));

    ROS_INFO("Dataset was successfully loaded from files\n%s\nand\n%s", std::string(packagePath + X_TRAIN_FILE_PATH).c_str(), std::string(packagePath + Y_TRAIN_FILE_PATH).c_str());
    ROS_INFO("Ready to predict");

    // Spinning mode
    ros::spin();
    return 0;
}
