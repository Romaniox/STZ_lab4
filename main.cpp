#include <iostream>
#include "custom_DFT.h"
#include "FourierTransform.h"


#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/imgproc.hpp>



int main() {
    cv::Mat img = cv::imread("./images/fourier.png");

    FourierTransform test(img);

    test.full_DFT_opencv(true, false, "../results/FFT_cpp_256_320.xml");

    return 0;
}
