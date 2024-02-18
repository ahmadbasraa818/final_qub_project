#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cmath>

using namespace cv;

std::tuple<Mat, Mat, Mat, Mat> processFFT(const Mat& img) {
    Mat img_gry;
    img.convertTo(img_gry, CV_32F);
    
    Mat f;
    dft(img_gry, f, DFT_COMPLEX_OUTPUT);

    Mat fshift;
    fshift = Mat(f.size(), f.type());
    int crow = f.rows / 2;
    int ccol = f.cols / 2;
    int d = 75; // Size of the rectangular region to be zeroed
    

    f(Rect(ccol - d/2, crow - d/2, d, d)) = Scalar::all(0);

    // Shift the zero frequency component back to the original position
    int cx = f.cols / 2;
    int cy = f.rows / 2;
    Mat q0(f, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(f, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(f, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(f, Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);

    Mat f_ishift;
    idft(f, f_ishift, DFT_REAL_OUTPUT | DFT_SCALE);

    Mat img_fft;
    f_ishift.convertTo(img_fft, CV_8U);

    // Calculate the magnitude spectrum
    Mat planes[] = {Mat::zeros(f.size(), CV_32F), Mat::zeros(f.size(), CV_32F)};
    split(f, planes);
    magnitude(planes[0], planes[1], planes[0]);
    Mat mag_spectrum = planes[0];

    return std::make_tuple(fshift, f_ishift, img_fft, mag_spectrum);
}

int main() {
    Mat img = imread("image.jpg", IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error: Image not found!\n";
        return -1;
    }

    Mat fshift, f_ishift, img_fft, mag_spectrum;
    std::tie(fshift, f_ishift, img_fft, mag_spectrum) = processFFT(img);

    // Display your results or do further processing
    imshow("FFT Shifted Image", img_fft);
    imshow("Magnitude Spectrum", mag_spectrum);
    waitKey(0);

    return 0;
}
