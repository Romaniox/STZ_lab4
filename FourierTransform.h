#include <iostream>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <opencv2/videoio.hpp>

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
    cv::Mat img;
    cv::Mat img_complex;
    cv::Mat img_back;
    cv::Mat padded_img;
    cv::Mat img_mag;
public:
    DFTType DFT_type  = DFT_OpenCV;
private:
//    cv::Mat padded_img;
//    cv::Mat img_mag;

public:
    int h;
    int w;
    int c;
public:
    int h1 = 0;
    int w1 = 0;

public:
    cv::Mat full_DFT(bool show = false, bool save = false, const std::string &save_path = "../results/result.xml");

    cv::Mat
    full_DFT_opencv(bool show = false, bool save = false, const std::string &save_path = "../results/result.xml");

    cv::Mat full_FFT(bool show = false, bool save = false, const std::string &save_path = "../results/result.xml");
//    cv::Mat full_FFT_opencv(bool show = false, bool save = false, const std::string& save_path = "../results/result.xml");
public:
    void dft(const cv::Mat &img_dft, cv::Mat &img_dft2, bool show = false, bool inverse = false);

    void idft(const cv::Mat &img_idft, cv::Mat &img_idft2, bool show = false);

    cv::Mat fft(const cv::Mat &x_in, int N, bool inv = false);

    void fft_rec(std::vector<std::complex<float>> &x, int N, bool inv = false);

    void ifft_rec(std::vector<std::complex<float>> &x, int N);

    cv::Mat ifft(const cv::Mat &x_in, int N);

public:
    cv::Mat DFT_lobovoy(const cv::Mat &x0, const cv::Mat &W);
private:
    void save(cv::Mat &imag, std::string save_path);

    void show_images(const std::vector<cv::Mat> &images);

private:
    cv::Mat get_W(int N);

    cv::Mat get_W_inv(int N);

public:
    void preproc();

    void postproc();


public:
    explicit FourierTransform(DFTType DFT_type  = DFT_OpenCV);

    ~FourierTransform() = default;

    cv::Mat only_back_DFT(bool show = false, bool save = false, const std::string &save_path = "../results/result.xml");

    void dir_DFT();

    void back_DFT();

    void full(bool show = false, bool save = false, const std::string &save_path = "../results/result.xml");

    void get_hwc();
};
