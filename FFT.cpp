#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <cmath>

using namespace std;
namespace fs = std::filesystem;

// Function to apply bluestiends effect
cv::Mat bluestiends(const cv::Mat& img) {
    // Convert the image to HSV color space
    cv::Mat result;
    cv::cvtColor(img, result, cv::COLOR_BGR2HSV);

    // Split the HSV image into separate channels
    std::vector<cv::Mat> channels;
    cv::split(result, channels);

    channels[0] = cv::Scalar(120);

    // Merge the channels back to obtain the modified image
    cv::merge(channels, result);

    // Convert the image back to BGR color space
    cv::cvtColor(result, result, cv::COLOR_HSV2BGR);

    return result;
}
// Function to perform FFT on an image
double fftApproximation(const cv::Mat& img) {
    // Convert the image to grayscale and float32
    cv::Mat imgGrayFloat32;
    img.convertTo(imgGrayFloat32, CV_32FC1);
    cv::cvtColor(imgGrayFloat32, imgGrayFloat32, cv::COLOR_BGR2GRAY);

    // Perform FFT
    cv::Mat complexImg;
    cv::dft(imgGrayFloat32, complexImg, cv::DFT_COMPLEX_OUTPUT);

    // Apply high-pass filter (remove low frequencies)
    int centerX = complexImg.cols / 2;
    int centerY = complexImg.rows / 2;
    int radius = 30;
    cv::circle(complexImg, cv::Point(centerX, centerY), radius, cv::Scalar(0, 0), -1);

    // Perform inverse FFT to get the approximation
    cv::Mat approxImg;
    cv::idft(complexImg, approxImg, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

    // Calculate the percentage of blur
    cv::Scalar mean, stddev;
    cv::meanStdDev(approxImg, mean, stddev);

    // Normalize the result
    cv::normalize(approxImg, approxImg, 0, 255, cv::NORM_MINMAX);

    // Convert the result back to uint8
    approxImg.convertTo(approxImg, CV_8UC1);

    // Calculate the percentage of the image that is blurry
    double percentageBlur = (stddev[0] / mean[0]) * 100.0;

    return percentageBlur;
}

int main(int argc, char* argv[]) {
    // Check if the correct number of command-line arguments is provided
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <file_path>" << endl;
        return 1;
    }

    const string filePath = argv[1];

    // Check if the file exists
    if (!fs::exists(filePath)) {
        cerr << "Error: The specified file does not exist." << endl;
        return 1;
    }

// Use the specified image file
    cv::Mat img = cv::imread(filePath);

    // Perform FFT approximation and calculate percentage of blur
    double percentageBlur = fftApproximation(img);

    // Apply bluestiends effect
    cv::Mat bluestiendsImg = bluestiends(img);

    // Display the original image, FFT approximation, bluestiends effect, and percentage of blur
    cout << "Percentage of Image That Is Blurry: " << percentageBlur << "%" << endl;

    cv::imshow("Original Image", img);
    cv::imshow("FFT Approximation", bluestiendsImg); // Display the bluestiends effect for clarity
    cv::waitKey(0);

    return 0;
}
