#include "khepera_hand_detect.hpp"

void khepera::extractHand(const cv::Mat& in, cv::Mat& out, unsigned char& seuil, cv::BackgroundSubtractor *pMOG2, bool& backgroundRemove) {

    cv::Mat frame, frame_original, foreground, kernel;

    frame_original = in.clone();

    unsigned int frameHeight = frame_original.rows;
    unsigned int frameWidth = frame_original.cols;
    unsigned int frameDim = frame_original.channels();

    //// BACKGROUND SUBTRACTOR ////
    if(backgroundRemove){
        //update the background model
        pMOG2->apply(frame_original, foreground, 0.001);
        // imshow("Foreground", foreground);

        //// FOREGROUND FILTERING ////
        kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, { 3, 3 });

        // Opening
        cv::morphologyEx(foreground, foreground, cv::MORPH_OPEN, kernel);
        // Fill any small holes
        cv::morphologyEx(foreground, foreground, cv::MORPH_CLOSE, kernel);
        // Remove noise
        cv::morphologyEx(foreground, foreground, cv::MORPH_OPEN, kernel);
        // Dilation
        cv::dilate(foreground, foreground, cv::Mat(), cv::Point(-1, -1), 3);

        frame_original.copyTo(frame, foreground);
        // putText(frame_original, "Background subtractor ON", cv::Point(15, 15),FONT_HERSHEY_SIMPLEX, 0.5 , cv::Scalar(0,0,255));
    } else {
        // putText(frame_original, "Background subtractor OFF", cv::Point(15, 15),FONT_HERSHEY_SIMPLEX, 0.5 , cv::Scalar(0,0,255));
        frame = frame_original.clone();
    }

    //// COLOR DETECTION ////
    cv::GaussianBlur(frame, frame, cv::Size(7,7), 1.5, 1.5);

    for (unsigned int index=0; index<frameHeight*frameWidth; index++)
    {
        bool detect=false;

        unsigned char R=frame.data[frameDim*index];
        unsigned char G=frame.data[frameDim*index + 1];
        unsigned char B=frame.data[frameDim*index + 2];

        if ((R>G) && (R>B))
            if (((R-B)>=seuil) && ((R-G)>=seuil))
                detect=true;

        if (detect==true){
            out.data[index]=255;
        } else{
            out.data[index]=0;
        }
    }
}
