
#include <cmath>
#include <complex>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/imgproc.hpp>

cv::Mat fft(const cv::Mat &x_in, int N);

void fft_rec(std::vector<std::complex<float>> &x, int N);
