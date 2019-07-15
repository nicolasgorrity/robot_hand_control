#pragma once
#include <complex>
#include "khepera_classification/Pixel.h"
#include "khepera_classification/Contour.h"

/*** PCA data ***/
#define COEFF_FILE_PATH "/files/pca_inv_coeffT.csv"
#define MU_FILE_PATH "/files/pca_mu.csv"

/*** Dataset ***/
#define X_TRAIN_FILE_PATH "/files/pca_Xtrain.csv"
#define Y_TRAIN_FILE_PATH "/files/pca_Ytrain.csv"

/*** Labels and messages ***/
#define FORWARD "Forward"
#define RIGHT "TurnRight"
#define LEFT "TurnLeft"
#define NONE "None"

#define FORWARD_LABEL 1
#define TURN_LABEL 2
#define STOP_LABEL 3
#define TRASH_LABEL 4

namespace khepera {

    void readContourMsg(const khepera_classification::Contour::ConstPtr& contourMsg, std::vector<std::complex<float>>& contour);

}
