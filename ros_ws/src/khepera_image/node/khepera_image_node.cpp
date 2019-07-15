#include "ros/ros.h"
#include <ros/package.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <dynamic_reconfigure/server.h>

#include <opencv2/opencv.hpp>

#include "khepera_camera.hpp"
#include "khepera_hand_detect.hpp"
#include "khepera_contour.hpp"

#include <khepera_image/ParamsConfig.h>
#include "khepera_image/Pixel.h"
#include "khepera_image/Contour.h"

#define DATABASE_PATH "/database/image_%02d.bmp"

// Callback of dynamic parameters configuration
void param_callback(khepera_image::ParamsConfig &config, uint32_t level, unsigned char& threshold, cv::VideoCapture& cap, bool& bgSubstractor, std::string& packagePath) {
    threshold = config.threshold;
    ROS_INFO("Parameter 'threshold' is configured to %d", threshold);

    if (config.camera_source) {
        if (cap.isOpened()) cap.release();
        cap = cv::VideoCapture(0);
        if (!khepera::initCamera(cap)) {
            ROS_ERROR("Could not initialize VideoCapture object to camera source %d", 0);
        } else {
            ROS_INFO("Video flux comes from camera %d", 0);
        }
    } else {
        if (cap.isOpened()) cap.release();
        cap = cv::VideoCapture(std::string(packagePath + DATABASE_PATH).c_str());
        if (!khepera::initCamera(cap)) {
            ROS_ERROR("Could not initialize VideoCapture object to images flux from database %s", std::string(packagePath + DATABASE_PATH).c_str());
        } else {
            ROS_INFO("Video flux comes from database %s", std::string(packagePath + DATABASE_PATH).c_str());
        }
    }

    bgSubstractor = config.bgSubstractor;
    ROS_INFO("Background substractor is %s", bgSubstractor ? "enabled" : "disabled");
}


int main(int argc, char **argv) {
    // Parameters
    unsigned char threshold = 40;
    bool bgSubstractor = false;

    // Initialization
    ros::init(argc, argv, "khepera_image");
    std::string packagePath = ros::package::getPath("khepera_image");
    ros::NodeHandle n;
    ros::Rate loop_rate(10);
    image_transport::ImageTransport it(n);

    // Tools
    cv::Ptr<cv::BackgroundSubtractor> pMOG2 = cv::createBackgroundSubtractorMOG2(); //MOG2 Background subtractor
    cv::VideoCapture cap;

    // Dynamic parameters server
    dynamic_reconfigure::Server<khepera_image::ParamsConfig> server;
    server.setCallback(boost::bind(&param_callback, _1, _2, boost::ref(threshold), boost::ref(cap), boost::ref(bgSubstractor), boost::ref(packagePath)));

    // Publishers of images
    image_transport::Publisher camera_publisher      = it.advertise("/khepera/camera"      , 1);
    image_transport::Publisher hand_publisher        = it.advertise("/khepera/hand_detect" , 1);
    image_transport::Publisher contour_img_publisher = it.advertise("/khepera/contours_img", 1);
    // Publisher of contour
    ros::Publisher contour_publisher = n.advertise<khepera_image::Contour>("/khepera/contours", 1);

    // Camera image
    cv::Mat camera_out;

    /*** LOOP ***/
    while (ros::ok()) {
        // Get image from camera
        khepera::getCameraImage(cap, camera_out);

        // Publish it
        if (!camera_out.empty()) {
            camera_publisher.publish(cv_bridge::CvImage(std_msgs::Header(), sensor_msgs::image_encodings::RGB8, camera_out).toImageMsg());

            // Extract hand from image
            cv::Mat hand_out(camera_out.rows, camera_out.cols, CV_8UC1);
            khepera::extractHand(camera_out, hand_out, threshold, pMOG2, bgSubstractor);
            hand_publisher.publish(cv_bridge::CvImage(std_msgs::Header(), sensor_msgs::image_encodings::MONO8, hand_out).toImageMsg());

            // Extract hand contours
            std::vector<cv::Point> contour;
            khepera::extractContour(hand_out, contour);

            // If a contour is found
            if (contour.size() > 0) {
                // Find contour orientation
                std::string orientation;
                khepera::findContourOrientation(contour, orientation);

                // Publish contour
                khepera_image::Contour contourMsg;
                for (auto it=contour.begin(); it!=contour.end(); ++it) {
                    khepera_image::Pixel pixel;
                    pixel.x = (*it).x;
                    pixel.y = (*it).y;
                    contourMsg.contour.push_back(pixel);
                }
                contourMsg.orientation = orientation;
                contour_publisher.publish(contourMsg);

                // Draw contour on output image and publish it
                cv::Mat contour_img(camera_out.rows, camera_out.cols, CV_8UC1, cv::Scalar(0));
                khepera::drawContour(contour_img, contour);
                contour_img_publisher.publish(cv_bridge::CvImage(std_msgs::Header(), sensor_msgs::image_encodings::MONO8, contour_img).toImageMsg());
            }
        }

        // Sleep
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
