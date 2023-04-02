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

    //x sobel
    cv::Mat kernel = (cv::Mat_<int>(3,3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
    //y sobel
//    cv::Mat kernel = (cv::Mat_<int>(3,3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);

    //laplas
//    cv::Mat kernel = (cv::Mat_<int>(3,3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);
    //box
//    cv::Mat kernel = (cv::Mat_<float>(3,3) << 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9);
//    cv::Mat kernel = (cv::Mat_<int>(3,3) << 1, 1, 1, 1, 1, 1, 1, 1, 1);



//    FourierTransform img_(img);
//    FourierTransform kernel_(kernel);
//    FourierTransform res(kernel);

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


//    cv::Mat kernel_fourier = kernel_.full_DFT_opencv(true, false);
//    cv::Mat img_fourier = img_.full_DFT_opencv(true, false, "../results/FFT_cpp_256_320.xml");
//
//    cv::Mat result;
//    cv::mulSpectrums(kernel_.img_complex, img_.img_complex, res.img_complex, 0);
//
//    res.only_back_DFT(true);

//    res.s
//    std::cout << res_f.img_back << std::endl;

    std::vector<cv::Mat> images = {init_img_f.img_mag, kernel_f.img_mag, res_f.img_back};
    show_images(images);

    return 0;
}
