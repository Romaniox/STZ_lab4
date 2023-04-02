#include <iostream>
#include "custom_DFT.h"
#include "FourierTransform.h"


#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/imgproc.hpp>

#define PART 8



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
/*

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

    std::vector<cv::Mat> images = {res_f.img_mag, res_f.img_back};
    show_images(images);

    */
    // part 9
    cv::Mat nomer = cv::imread("./images/nomer.png", cv::IMREAD_GRAYSCALE);
    cv::Mat tpl = cv::imread("./images/nomer2.png", cv::IMREAD_GRAYSCALE);

//    cv::Mat nomer = cv::imread("./images/Number.jpeg", cv::IMREAD_GRAYSCALE);
//    cv::Mat tpl = cv::imread("./images/A.jpeg", cv::IMREAD_GRAYSCALE);

    nomer = 255 - nomer;
    tpl = 255 - tpl;

    FourierTransform nomer_f;
    FourierTransform tpl_f;

    nomer_f.img = nomer;
    tpl_f.img = tpl;

    nomer_f.full(DFT);

    tpl_f.w1 = nomer_f.w1;
    tpl_f.h1 = nomer_f.h1;
    tpl_f.full(DFT);

    FourierTransform res_f;
    res_f.w = nomer_f.w;
    res_f.h = nomer_f.h;
    res_f.w1 = nomer_f.w1;
    res_f.h1 = nomer_f.h1;

    cv::mulSpectrums(nomer_f.img_complex, tpl_f.img_complex, res_f.img_complex, 0, true);
    res_f.back_DFT(DFT);
    res_f.postproc();

    cv::Mat result = res_f.img_back.clone();

    double minVal;
    double maxVal;
    cv::minMaxLoc(result, &minVal, &maxVal);
    std::cout << maxVal << std::endl;

    cv::threshold(result, result,  maxVal - 0.1, maxVal, cv::THRESH_BINARY);


    std::vector<cv::Mat> images = {nomer_f.img_back, tpl_f.img_back, res_f.img_back, result};
    show_images(images);

    return 0;
}
