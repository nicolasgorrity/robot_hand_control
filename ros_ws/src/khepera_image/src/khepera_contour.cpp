#include "khepera_contour.hpp"

#define RIGHT "TurnRight"
#define LEFT "TurnLeft"

void khepera::extractContour(const cv::Mat& in, std::vector<cv::Point>& contour) {
    // Structures to fill
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    // Extract all contours in image
    findContours(in, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    // Find biggest contour
    unsigned int max_contour = 0;
    unsigned int idx_max_contour = -1;

    for (unsigned int numc=0; numc<contours.size(); ++numc) {
        if (cv::contourArea(contours[numc]) > max_contour) {
            idx_max_contour = numc;
            max_contour = cv::contourArea(contours[numc]);
        }
    }

    if (idx_max_contour != -1) contour = std::move(contours[idx_max_contour]);
}

void khepera::findContourOrientation(std::vector<cv::Point>& contour, std::string& orientation) {
    cv::RotatedRect rect = cv::minAreaRect(contour);
	float angle = rect.angle + 90;
	if (rect.size.width < rect.size.height) {
		angle += 90;
	}

	if (angle>=0 && angle<90) orientation = LEFT;
	else if (angle>=90 && angle<=180) orientation = RIGHT;
    else orientation = "";
}

void khepera::drawContour(cv::Mat& out, std::vector<cv::Point>& contour) {
    std::vector<std::vector<cv::Point>> contoursToDraw;
    contoursToDraw.push_back(contour);
    drawContours(out, contoursToDraw, 0, 255);;
}
