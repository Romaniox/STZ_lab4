#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#include <iostream>

enum DFTType {
    DFT,
    DFT_OpenCV,
    FFT
};

enum FilterType {
    XSobel,
    YSobel,
    Laplas,
    Box
};

void show_images(const std::vector<cv::Mat> &images);
void krasivSpektr(cv::Mat &imag);

class FourierTransform {
public:
    cv::Mat img0;
    cv::Mat img_complex;
    cv::Mat img_back;
    cv::Mat img_padded;
    cv::Mat img_mag;
public:
    DFTType DFT_type  = DFT_OpenCV;
public:
    int h = 0;
    int w = 0;
    int c = 0;
public:
    int h1 = 0;
    int w1 = 0;
public:
    void preproc();
    void postproc();
    bool normalize = true;
private:
    void dft(const cv::Mat &img_dft, cv::Mat &img_dft2, bool show = false, bool inverse = false);
    void idft(const cv::Mat &img_idft, cv::Mat &img_idft2, bool show = false);
    cv::Mat DFT_lobovoy(const cv::Mat &x0, const cv::Mat &W);

    void fft(const cv::Mat &img_dft);
    void ifft(const cv::Mat &img_dft, const cv::Mat &img_dft2);
    cv::Mat fft_(const cv::Mat &x_in, int N, bool inv = false);
    cv::Mat ifft_(const cv::Mat &x_in, int N);
    void fft_rec(std::vector<std::complex<float>> &x, int N, bool inv = false);
    void ifft_rec(std::vector<std::complex<float>> &x, int N);
private:
    cv::Mat get_W(int N);
    cv::Mat get_W_inv(int N);
    void get_hwc();
private:
    void save(cv::Mat &imag, std::string save_path);
    void show_images(const std::vector<cv::Mat> &images);
public:
    explicit FourierTransform(DFTType DFT_type  = DFT_OpenCV);
    ~FourierTransform() = default;

    void dir_DFT();
    void back_DFT();
    void full(bool show = false, bool save = false, const std::string &save_path = "../results/result.xml");
};
