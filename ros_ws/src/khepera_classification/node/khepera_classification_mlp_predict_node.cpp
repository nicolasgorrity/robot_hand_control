#include "ros/ros.h"
#include <ros/package.h>
#include <std_msgs/String.h>
#include <opencv2/opencv.hpp>

#include "khepera_classification.hpp"
#include "khepera_fourier_transform.hpp"
#include "khepera_pca.hpp"
#include "khepera_classif_mlp.hpp"

void callback(const boost::shared_ptr<khepera_classification::Contour const> msg,
              std::vector<std::vector<float>>& pcaCoefficients,
              std::vector<float>& pcaMu,
              Ptr<ANN_MLP>& network,
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
        Mat result;
        khepera::predictWithMLP(network, X_pca, result);
        unsigned short predictedLabel = khepera::retrieveMlpPrediction(result);

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

    // Load  classifier (only needs to be done once)
    Ptr<ANN_MLP> network;
    khepera::loadMultiLayerPerceptron(packagePath + MLP_FILE_PATH, network);

    // Subscriber of contour messages
    ros::Subscriber sub = n.subscribe<khepera_classification::Contour>("khepera/contours", 1000,
                                boost::bind(callback, _1,
                                            boost::ref(pcaCoefficients), boost::ref(pcaMu),
                                            boost::ref(network),
                                            boost::ref(commands_publisher)));

    ROS_INFO("Neural network was successfully loaded from %s", std::string(packagePath + MLP_FILE_PATH).c_str());
    ROS_INFO("Ready to predict");

    // Spinning mode
    ros::spin();
    return 0;
}
