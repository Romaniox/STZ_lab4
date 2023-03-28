#include "custom_FFT.h"
#include <iostream>

cv::Mat fft(const cv::Mat &x_in, int N) {
    std::vector<std::complex<float>> x_out(N);

//    if (N == 512) {
//        std::cout << x_in.at<cv::Point2f>(0, 0) << std::endl;
//        std::cout << x_in.at<cv::Point2f>(0, 100) << std::endl;
//        std::cout << x_in.at<cv::Point2f>(0, 50) << std::endl;
//        std::cout << x_in.at<cv::Point2f>(0, 250) << std::endl;
//    }


//    std::complex<float> *x_out;
    // Make copy of array and apply window
//    std::cout << x_in.type() << std::endl;
    for (int i = 0; i < N; i++) {
//        std::cout << x_in.at<float>(0, 1) << std::endl;
//        std::cout << x_in.at<cv::Point2f>(0, i).x << std::endl;
//        std::cout << x_in.at<cv::Point2f>(0, i).y << std::endl;

        x_out[i] = std::complex<float>(x_in.at<cv::Point2f>(0, i).x, x_in.at<cv::Point2f>(0, i).y);
        x_out[i] *= 1; // Window
    }

    // Start recursion
    fft_rec(x_out, N);

    cv::Mat x_out_(x_in.rows, x_in.cols, CV_32FC2);
    for (int i = 0; i < x_out_.cols; i++) {
//        std::cout << x_out[i].real() << std::endl;
//        std::cout << x_out[i].imag() << std::endl;
//        std::cout << cv::Point2f (x_out[i].real(), x_out[i].imag()) << std::endl;

        x_out_.at<cv::Point2f>(0, i) = cv::Point2f(x_out[i].real(), x_out[i].imag());
//        std::cout << x_out_.at<cv::Point2f>(0, i) << std::endl;
    }

//    cv::merge(std::vector<cv::Mat>{x_out->real(), x_out->imag()}, W);

//    std::cout << x_out_.at<cv::Point2f>(0, 100) << std::endl;

    return x_out_;
}

void fft_rec(std::vector<std::complex<float>> &x, int N) {
    // Check if it is splitted enough
    if (N <= 1) {
        return;
    }

    // Split even and odd
    std::vector<std::complex<float>> odd(N / 2);
    std::vector<std::complex<float>> even(N / 2);
    for (int i = 0; i < N / 2; i++) {
        even[i] = x[i * 2];
        odd[i] = x[i * 2 + 1];
    }

    // Split on tasks
    fft_rec(even, N / 2);
    fft_rec(odd, N / 2);


    // Calculate DFT
    for (int k = 0; k < N / 2; k++) {
        std::complex<float> W = exp(std::complex<float>(0, -2 * CV_PI * k / N));

        x[k] = even[k] + W * odd[k];
        x[N / 2 + k] = even[k] - W * odd[k];
    }
}