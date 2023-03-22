#include "custom_DFT.h"
#include <cmath>

cv::Mat get_W(int N) {
    // get n and k vectors
    cv::Mat n(1, N, CV_32F);
    for (int i = 0; i < N; i++) {
        n.at<float>(0, i) = (float) i;
    }
    cv::Mat k;
    cv::transpose(n, k);

    // realize from Python: W = np.exp(-2j * np.pi * k * n / N)
    cv::Mat W;
    W = k * n;
    cv::divide(W, N, W);
    cv::multiply(CV_PI, W, W);

    // create Re/Im matrix of W (W_r, W_i)
    cv::Mat tmp = W * -2;
    cv::Mat Im_(N, N, CV_32FC2);;
    cv::merge(std::vector<cv::Mat>{cv::Mat::zeros(W.size(), CV_32F), tmp}, Im_);

    cv::Mat W_r(N, N, CV_32F);
    cv::Mat W_i(N, N, CV_32F);
    for (int i = 0; i < Im_.rows; i++) {
        for (int j = 0; j < Im_.cols; j++) {
            W_r.at<float>(i, j) = cos(Im_.at<cv::Point2f>(i, j).y);
            W_i.at<float>(i, j) = sin(Im_.at<cv::Point2f>(i, j).y);
        }
    }
    cv::merge(std::vector<cv::Mat>{W_r, W_i}, W);

    return W;
}

cv::Mat get_W_inv(int N) {
    // get n and k vectors
    cv::Mat n(1, N, CV_32F);
    for (int i = 0; i < N; i++) {
        n.at<float>(0, i) = (float) i;
    }
    cv::Mat k;
    cv::transpose(n, k);

    // realize from Python: W = np.exp(2j * np.pi * k * n / N) / N
    cv::Mat W;
    W = k * n;
    cv::divide(W, N, W);
    cv::multiply(CV_PI, W, W);

    // create Re/Im matrix of W (W_r, W_i)
    cv::Mat tmp = W * 2;
    cv::Mat Im_(N, N, CV_32FC2);;
    cv::merge(std::vector<cv::Mat>{cv::Mat::zeros(W.size(), CV_32F), tmp}, Im_);

    cv::Mat W_r(N, N, CV_32F);
    cv::Mat W_i(N, N, CV_32F);
    for (int i = 0; i < Im_.rows; i++) {
        for (int j = 0; j < Im_.cols; j++) {
            W_r.at<float>(i, j) = cos(Im_.at<cv::Point2f>(i, j).y);
            W_i.at<float>(i, j) = sin(Im_.at<cv::Point2f>(i, j).y);
        }
    }
    cv::merge(std::vector<cv::Mat>{W_r, W_i}, W);

    W /= N;
    return W;
}

cv::Mat DFT_lobovoy(const cv::Mat &x0, const cv::Mat &W) {

    // edit input vector x
    cv::Mat x_out;
    if (x0.type() == CV_32FC1) {
        cv::merge(std::vector<cv::Mat>{x0, cv::Mat::zeros(x0.size(), CV_32F)}, x_out);
    } else {
        x_out = x0;
    }

    if (x_out.rows == 1) {
        cv::transpose(x_out, x_out);
    }

    // get output vector X
    cv::Mat X = W * x_out;
    return X;
}
