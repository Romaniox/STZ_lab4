#include <iostream>
#include "custom_DFT.h"
#include "FourierTransform.h"


#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/imgproc.hpp>



int main() {
    cv::Mat img = cv::imread("./images/fourier.png");

//    int data[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
//    cv::Mat kernel = cv::Mat(3, 3, CV_8SC1);

    //part 7

    //x sobel
//    cv::Mat kernel = (cv::Mat_<int>(3,3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
    //y sobel
//    cv::Mat kernel = (cv::Mat_<int>(3,3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);

    //laplas
//    cv::Mat kernel = (cv::Mat_<int>(3,3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);
    //box
//    cv::Mat kernel = (cv::Mat_<float>(3,3) << 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9);
//    cv::Mat kernel = (cv::Mat_<int>(3,3) << 1, 1, 1, 1, 1, 1, 1, 1, 1);

/*
    FourierTransform init_img_f;
    init_img_f.img = img;

    init_img_f.full(DFT, false, false);

    FourierTransform kernel_f;
    kernel_f.img = kernel;
    kernel_f.full(DFT);

    FourierTransform res_f;
    cv::mulSpectrums(init_img_f.img_complex, kernel_f.img_complex, res_f.img_complex, 0);

    res_f.back_DFT(DFT);
    res_f.postproc();

    std::vector<cv::Mat> images = {init_img_f.img_mag, kernel_f.img_mag, res_f.img_back};
    show_images(images);
*/


    //part 8


    FourierTransform init_img_f;
    init_img_f.img = img;

    init_img_f.full(DFT, false, false);

    // low pass filter
//    cv::Mat mask = cv::Mat(init_img_f.img_mag.rows, init_img_f.img_mag.cols, CV_8UC1, cv::Scalar(0));
//    cv::circle(mask, cv::Point(mask.cols / 2, mask.rows / 2), 100, cv::Scalar(255), -1);
//    krasivSpektr(mask);

    // high pass filter
    cv::Mat mask = cv::Mat(init_img_f.img_mag.rows, init_img_f.img_mag.cols, CV_8UC1, cv::Scalar(255));
    cv::circle(mask, cv::Point(mask.cols / 2, mask.rows / 2), 20, cv::Scalar(0), -1);
    krasivSpektr(mask);


    FourierTransform res_f;
    cv::bitwise_and(init_img_f.img_complex, init_img_f.img_complex, res_f.img_complex, mask);
//    cv::mulSpectrums(init_img_f.img_complex, mask_f.img_complex, res_f.img_complex, 0);

    res_f.back_DFT(DFT);
    res_f.postproc();

//    cv::Mat out;
//    std::cout << init_img_f.img_mag.type() << std::endl;
//    cv::bitwise_and(init_img_f.img_mag, init_img_f.img_mag, out, mask);



//    FourierTransform kernel_f;
//    kernel_f.img = kernel;
//    kernel_f.full(DFT);
//
//    FourierTransform res_f;
//    cv::mulSpectrums(init_img_f.img_complex, kernel_f.img_complex, res_f.img_complex, 0);
//
//    res_f.back_DFT(DFT);
//    res_f.postproc();

    std::vector<cv::Mat> images = {res_f.img_mag, res_f.img_back};
    show_images(images);

    return 0;
}
