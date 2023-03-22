#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/imgproc.hpp>
#include <iostream>

cv::Mat DFT_lobovoy(const cv::Mat& x_, const cv::Mat &W);
cv::Mat get_W(int N);
cv::Mat get_W_inv(int N);