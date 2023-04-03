#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"

#include "FourierTransform.h"

#define PART9

DFTType DFT_TYPE = DFT;

int main() {
    cv::Mat img = cv::imread("./images/fourier.png");

#ifdef PART1
    FourierTransform init_img_f(DFT_TYPE);
    init_img_f.img0 = img;

    init_img_f.full();

    std::vector<cv::Mat> images = {init_img_f.img0, init_img_f.img_padded, init_img_f.img_mag, init_img_f.img_back};
    show_images(images);
#endif

#ifdef PART7
    FilterType filter = Box;

    cv::Mat kernel;
    switch (filter) {
        case XSobel:
            kernel = (cv::Mat_<int>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
            break;
        case YSobel:
            kernel = (cv::Mat_<int>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
            break;
        case Laplas:
            kernel = (cv::Mat_<int>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);
            break;
        case Box:
            kernel = (cv::Mat_<int>(3, 3) << 1, 1, 1, 1, 1, 1, 1, 1, 1);
            break;
    }

    FourierTransform init_img_f(DFT_TYPE);
    init_img_f.img0 = img;

    init_img_f.full();

    FourierTransform kernel_f(DFT_TYPE);
    kernel_f.img0 = kernel;
    kernel_f.h1 = init_img_f.h1;
    kernel_f.w1 = init_img_f.w1;

    kernel_f.full();

    FourierTransform res_f(DFT_TYPE);
    res_f.h = init_img_f.h;
    res_f.w = init_img_f.w;

    cv::mulSpectrums(init_img_f.img_complex, kernel_f.img_complex, res_f.img_complex, 0, false);

    res_f.back_DFT();
    res_f.postproc();

    std::vector<cv::Mat> images = {init_img_f.img_padded, init_img_f.img_mag, kernel_f.img_mag, res_f.img_mag, res_f.img_back};
    show_images(images);
#endif

#ifdef PART8
    FourierTransform init_img_f(DFT_TYPE);
    init_img_f.img0 = img;

    init_img_f.full();

    // low pass filter
    cv::Mat mask_lpf = cv::Mat(init_img_f.img_mag.rows, init_img_f.img_mag.cols, CV_8UC1, cv::Scalar(0));
    cv::circle(mask_lpf, cv::Point(mask_lpf.cols / 2, mask_lpf.rows / 2), 50, cv::Scalar(255), -1);
    krasivSpektr(mask_lpf);

    // high pass filter
    cv::Mat mask_hpf = cv::Mat(init_img_f.img_mag.rows, init_img_f.img_mag.cols, CV_8UC1, cv::Scalar(255));
    cv::circle(mask_hpf, cv::Point(mask_hpf.cols / 2, mask_hpf.rows / 2), 20, cv::Scalar(0), -1);
    krasivSpektr(mask_hpf);

    cv::Mat mask = mask_hpf;

    FourierTransform res_f(DFT_TYPE);
    res_f.w = init_img_f.w;
    res_f.h = init_img_f.h;
    res_f.w1 = init_img_f.w1;
    res_f.h1 = init_img_f.h1;

    cv::bitwise_and(init_img_f.img_complex, init_img_f.img_complex, res_f.img_complex, mask);

    res_f.back_DFT();
    res_f.postproc();

    krasivSpektr(mask);
    std::vector<cv::Mat> images = {init_img_f.img0, init_img_f.img_mag, mask, res_f.img_mag, res_f.img_back};
    show_images(images);
#endif

#ifdef PART9
    cv::Mat nomer = cv::imread("./images/nomer.png", cv::IMREAD_GRAYSCALE);
    cv::Mat tpl = cv::imread("./images/nomer2.png", cv::IMREAD_GRAYSCALE);

    // inverse
    nomer = 255 - nomer;
    tpl = 255 - tpl;

    FourierTransform nomer_f(DFT_TYPE);
    FourierTransform tpl_f(DFT_TYPE);

    nomer_f.img0 = nomer;
    tpl_f.img0 = tpl;

    nomer_f.full(DFT);

    tpl_f.w1 = nomer_f.w1;
    tpl_f.h1 = nomer_f.h1;
    tpl_f.full(DFT);

    FourierTransform res_f(DFT_TYPE);
    res_f.w = nomer_f.w;
    res_f.h = nomer_f.h;
    res_f.w1 = nomer_f.w1;
    res_f.h1 = nomer_f.h1;

    cv::mulSpectrums(nomer_f.img_complex, tpl_f.img_complex, res_f.img_complex, 0, true);
    res_f.back_DFT();
    res_f.postproc();

    cv::Mat result = res_f.img_back.clone();

    double minVal;
    double maxVal;
    cv::minMaxLoc(result, &minVal, &maxVal);

    cv::threshold(result, result,  maxVal - 0.1, maxVal, cv::THRESH_BINARY);

    std::vector<cv::Mat> images = {nomer_f.img_back, tpl_f.img_back, res_f.img_back, result};
    show_images(images);
#endif

    return 0;
}
