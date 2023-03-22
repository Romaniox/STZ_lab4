#include <iostream>
#include "custom_DFT.h"

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/imgproc.hpp>

void krasivSpektr(cv::Mat &magI) {
    int cx = magI.cols / 2;
    int cy = magI.rows / 2;

    cv::Mat q0(magI, cv::Rect(0, 0, cx, cy)); // Top-Left - Create a ROI per quadrant
    cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy)); // Top-Right
    cv::Mat q2(magI, cv::Rect(0, cy, cx, cy)); // Bottom-Left
    cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

    cv::Mat tmp; // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp); // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

void show_images(const std::vector<cv::Mat> &images) {
    for (int i = 0; i < images.size(); i++) {
        cv::imshow(std::to_string(i), images[i]);
    }
    cv::waitKey(-1);
}

int main() {
    cv::Mat img = cv::imread("./images/fourier.png");
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

    cv::Mat padded;                            //expand input image to optimal size
    int m = cv::getOptimalDFTSize(img.rows);
    int n = cv::getOptimalDFTSize(img.cols);
    copyMakeBorder(img, padded, 0, m - img.rows, 0, n - img.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    cv::Mat padded_float = cv::Mat_<float>(padded);
    cv::Mat img_complex (padded_float.rows, padded_float.cols, CV_32FC2);

    // Fourier transform for every row in init img
    cv::Mat W;
    W = get_W(padded_float.cols);
    for (int i = 0; i < padded_float.rows; i++) {
        cv::Mat row = padded_float.row(i);

        cv::Mat row_new = DFT_lobovoy(row, W);
        cv::transpose(row_new, row_new);
        img_complex.row(i) = row_new.clone() + 0;
    }

    // Fourier transform for every col in edited img
    W = get_W(padded_float.rows);
    for (int i = 0; i < img_complex.cols; i++) {
        cv::Mat col = img_complex.col(i);
        cv::Mat col_new = DFT_lobovoy(col, W);
        img_complex.col(i) = col_new.clone() + 0;
    }

    //split Re and Im parts of Fourier img
    cv::Mat planes[2];
    split(img_complex, planes);

    // get visible Fourier result img
    cv::Mat img_mag;
    magnitude(planes[0], planes[1], img_mag);

    img_mag += cv::Scalar::all(1);
    cv::log(img_mag, img_mag);
    cv::normalize(img_mag, img_mag, 0, 1, cv::NORM_MINMAX);

    krasivSpektr(img_mag);

    // get init img by Inverse DFT
    cv::Mat img_back(img_complex.rows, img_complex.cols, CV_32FC2);

    cv::Mat W_inv;
    W_inv = get_W_inv(img_complex.cols);
    for (int i = 0; i < img_complex.rows; i++) {
        cv::Mat row = img_complex.row(i);

        cv::Mat row_new = DFT_lobovoy(row, W_inv);
        cv::transpose(row_new, row_new);
        img_back.row(i) = row_new.clone() + 0;
    }

    // Fourier transform for every col in edited img
    W_inv = get_W_inv(img_complex.rows);
    for (int i = 0; i < img_back.cols; i++) {
        cv::Mat col = img_back.col(i);
        cv::Mat col_new = DFT_lobovoy(col, W_inv);
        img_back.col(i) = col_new.clone() + 0;
    }

//    std::cout << img_back.at<cv::Point2f>(0, 0) << std::endl;
//    std::cout << img_back.at<cv::Point2f>(0, 1) << std::endl;
//    std::cout << img_back.at<cv::Point2f>(0, 2) << std::endl;

    cv::Mat planes_out[2];
    split(img_back, planes_out);

//    cv::Mat img_back_new;
    img_back = planes_out[0].clone();

//    std::cout << img_back << std::endl;

    cv::normalize(img_back, img_back, 0, 255, cv::NORM_MINMAX);
    img_back.convertTo(img_back, CV_8U);

    std::vector<cv::Mat> images = {img, padded, img_mag, img_back};
    show_images(images);

    return 0;
}
