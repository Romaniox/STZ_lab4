#include "FourierTransform.h"
#include <chrono>


void show_images(const std::vector<cv::Mat> &images) {
    for (int i = 0; i < images.size(); i++) {
        cv::imshow(std::to_string(i), images[i]);
    }
    cv::waitKey(-1);
}

void krasivSpektr(cv::Mat &imag) {
    int cx = imag.cols / 2;
    int cy = imag.rows / 2;

    cv::Mat q0(imag, cv::Rect(0, 0, cx, cy)); // Top-Left - Create a ROI per quadrant
    cv::Mat q1(imag, cv::Rect(cx, 0, cx, cy)); // Top-Right
    cv::Mat q2(imag, cv::Rect(0, cy, cx, cy)); // Bottom-Left
    cv::Mat q3(imag, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

    cv::Mat tmp; // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp); // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

FourierTransform::FourierTransform(DFTType DFT_type) {
    this->DFT_type = DFT_type;
}

void FourierTransform::get_hwc() {
    this->h = this->img0.rows;
    this->w = this->img0.cols;
    this->c = this->img0.channels();
}

// create padded and empty complex imgs
void FourierTransform::preproc() {
    get_hwc();
    if (this->c != 1) {
        cv::cvtColor(this->img0, this->img0, cv::COLOR_BGR2GRAY);
    }

    //expand input image to optimal size
    cv::Mat padded;
    if (this->h1 == 0 || this->w1 == 0) {
        if (this->DFT_type != FFT) {
            this->h1 = cv::getOptimalDFTSize(this->h);
            this->w1 = cv::getOptimalDFTSize(this->w);
        } else {
            this->h1 = 512;
            this->w1 = 256;
        }
    }

    copyMakeBorder(this->img0, padded, 0, this->h1 - this->h, 0, this->w1 - this->w, cv::BORDER_CONSTANT,
                   cv::Scalar::all(0));

    padded = cv::Mat_<float>(padded);
    this->img_padded = padded.clone();
    this->img_complex = cv::Mat(this->img_padded.rows, this->img_padded.cols, CV_32FC2);
}

// create beautiful magnitude and back imgs
void FourierTransform::postproc() {
    //split Re and Im parts of Fourier img
    cv::Mat planes[2];
    cv::split(this->img_complex, planes);

    // get visible Fourier result img
    magnitude(planes[0], planes[1], this->img_mag);

    this->img_mag += cv::Scalar::all(1);
    cv::log(this->img_mag, this->img_mag);
    cv::normalize(this->img_mag, this->img_mag, 0, 1, cv::NORM_MINMAX);

    krasivSpektr(this->img_mag);

    cv::Mat planes_out[2];
    split(this->img_back, planes_out);

    this->img_back = planes_out[0].clone();
    cv::normalize(this->img_back, this->img_back, 0, 255, cv::NORM_MINMAX);

    this->img_back.convertTo(this->img_back, CV_8U);
    this->img_back = this->img_back(cv::Rect(0, 0, this->w, this->h));

    this->img_padded.convertTo(this->img_padded, CV_8U);
}

// direct Fourier transform
void FourierTransform::dir_DFT() {
    cv::merge(std::vector<cv::Mat>{this->img_padded, cv::Mat::zeros(this->img_padded.size(), CV_32F)},
              this->img_complex);

//    auto start = std::chrono::high_resolution_clock::now();
//    auto end = std::chrono::high_resolution_clock::now();
//    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    switch (this->DFT_type) {
        case DFT:
            this->dft(this->img_padded, this->img_complex);
            break;

        case DFT_OpenCV:
            cv::dft(this->img_complex, this->img_complex);
            break;

        case FFT:
            this->fft(this->img_complex);
            break;
    }
}

// back Fourier transform
void FourierTransform::back_DFT() {
    this->img_back = cv::Mat(this->img_complex.rows, this->img_complex.cols, CV_32FC2);

    switch (this->DFT_type) {
        case DFT:
            this->idft(this->img_complex, this->img_back);
            break;

        case DFT_OpenCV:
            cv::dft(this->img_complex, this->img_back, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
            break;

        case FFT:
            this->ifft(this->img_complex, this->img_back);
            break;
    }
}

// full Fourier transform + back steps
void FourierTransform::full(bool show, bool save, const std::string &save_path) {
    this->preproc();
    this->dir_DFT();
    this->back_DFT();
    this->postproc();

    if (save) {
        this->save(this->img_complex, save_path);
    }

    if (show) {
        std::vector<cv::Mat> images = {this->img_mag};
        this->show_images(images);
    }
}

void FourierTransform::dft(const cv::Mat &img_dft, cv::Mat &img_dft2, bool show, bool inverse) {
    // Fourier transform for every row in init img
    cv::Mat W;
    W = this->get_W(img_dft.cols);
    for (int i = 0; i < img_dft.rows; i++) {
        cv::Mat row = img_dft.row(i);
        cv::Mat row_new = DFT_lobovoy(row, W);
        cv::transpose(row_new, row_new);
        img_dft2.row(i) = row_new.clone() + 0;
    }

    // Fourier transform for every col in edited img
    W = this->get_W(img_dft.rows);
    for (int i = 0; i < img_dft2.cols; i++) {
        cv::Mat col = img_dft2.col(i);
        cv::Mat col_new = DFT_lobovoy(col, W);
        img_dft2.col(i) = col_new.clone() + 0;
    }
}

void FourierTransform::idft(const cv::Mat &img_idft, cv::Mat &img_idft2, bool show) {
    cv::Mat W;
    W = this->get_W_inv(img_idft.cols);
    for (int i = 0; i < img_idft.rows; i++) {
        cv::Mat row = img_idft.row(i);

        cv::Mat row_new = DFT_lobovoy(row, W);
        cv::transpose(row_new, row_new);
        img_idft2.row(i) = row_new.clone() + 0;
    }

    // Fourier transform for every col in edited img
    W = this->get_W_inv(img_idft.rows);
    for (int i = 0; i < img_idft2.cols; i++) {
        cv::Mat col = img_idft2.col(i);
        cv::Mat col_new = DFT_lobovoy(col, W);
        img_idft2.col(i) = col_new.clone() + 0;
    }
}

cv::Mat FourierTransform::DFT_lobovoy(const cv::Mat &x0, const cv::Mat &W) {
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


cv::Mat FourierTransform::get_W(int N) {
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

cv::Mat FourierTransform::get_W_inv(int N) {
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

void FourierTransform::fft(const cv::Mat &img_dft) {
    for (int i = 0; i < img_dft.rows; i++) {
        cv::Mat row = img_dft.row(i);
        cv::Mat row_new = this->fft_(row, img_dft.cols);
        img_dft.row(i) = row_new.clone() + 0;
    }

    for (int i = 0; i < img_dft.cols; i++) {
        cv::Mat col = img_dft.col(i);
        cv::transpose(col, col);
        cv::Mat col_new = this->fft_(col, img_dft.rows);
        cv::transpose(col_new, col_new);
        img_dft.col(i) = col_new.clone() + 0;
    }
}

void FourierTransform::ifft(const cv::Mat &img_dft, const cv::Mat &img_dft2) {
    for (int i = 0; i < img_dft2.rows; i++) {
        cv::Mat row = img_dft.row(i);
        cv::Mat row_new = this->ifft_(row, img_dft.cols);
        img_dft2.row(i) = row_new.clone() + 0;
    }

    for (int i = 0; i < img_dft2.cols; i++) {
        cv::Mat col = img_dft2.col(i);
        cv::transpose(col, col);
        cv::Mat col_new = this->ifft_(col, img_dft2.rows);
        cv::transpose(col_new, col_new);
        img_dft2.col(i) = col_new.clone() + 0;
    }
}


cv::Mat FourierTransform::fft_(const cv::Mat &x_in, int N, bool inv) {
    std::vector<std::complex<float>> x_out(N);

    for (int i = 0; i < N; i++) {
        x_out[i] = std::complex<float>(x_in.at<cv::Point2f>(0, i).x, x_in.at<cv::Point2f>(0, i).y);
    }

    // Start recursion
    this->fft_rec(x_out, N, inv);

    cv::Mat x_out_(x_in.rows, x_in.cols, CV_32FC2);
    for (int i = 0; i < x_out_.cols; i++) {
        x_out_.at<cv::Point2f>(0, i) = cv::Point2f(x_out[i].real(), x_out[i].imag());
    }
    return x_out_;
}

cv::Mat FourierTransform::ifft_(const cv::Mat &x_in, int N) {
    cv::Mat x_out_ = fft_(x_in, N, true);
    return x_out_;
}

void FourierTransform::fft_rec(std::vector<std::complex<float>> &x, int N, bool inv) {
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
    fft_rec(even, N / 2, inv);
    fft_rec(odd, N / 2, inv);

    // Calculate DFT
    for (int k = 0; k < N / 2; k++) {
        std::complex<float> W;
        if (!inv) {
            W = exp(std::complex<float>(0, -2 * CV_PI * k / N));
        } else {
            W = exp(std::complex<float>(0, 2 * CV_PI * k / N));
        }

        x[k] = even[k] + W * odd[k];
        x[N / 2 + k] = even[k] - W * odd[k];
    }
}

void FourierTransform::ifft_rec(std::vector<std::complex<float>> &x, int N) {
    fft_rec(x, N, true);
}

void FourierTransform::save(cv::Mat &imag, std::string save_path) {
    cv::FileStorage file(save_path, cv::FileStorage::WRITE);
    file << "Mat" << imag;
}

void FourierTransform::show_images(const std::vector<cv::Mat> &images) {
    for (int i = 0; i < images.size(); i++) {
        cv::imshow(std::to_string(i), images[i]);
    }
    cv::waitKey(-1);
}
