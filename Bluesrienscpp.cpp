#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/slic/slic.hpp>

// Function to obtain superpixel masks using SLIC segmentation
std::vector<cv::Mat> getMasks(const cv::Mat& img, int n_seg = 250) {
    cv::Ptr<cv::ximgproc::SuperpixelSLIC> slic = cv::ximgproc::createSuperpixelSLIC(img, cv::ximgproc::SLIC, n_seg, 10.0f, 1.0f);
    slic->iterate(10);
    slic->getLabels(labels);

    std::vector<cv::Mat> masks;
    for (int i = 0; i < n_seg; ++i) {
        cv::Mat mask = cv::Mat::zeros(img.size(), CV_8U);
        cv::compare(labels, i, mask, cv::CMP_EQ);
        masks.push_back(mask);
    }

    return masks;
}

// Function for morphological operations on a mask
cv::Mat morphology(const cv::Mat& msk) {
    assert(msk.type() == CV_8U, "msk must be a grayscale image");

    // Erosion operation
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::erode(msk, msk, kernel, cv::Point(-1, -1), 1);

    // Closing operation
    kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(msk, msk, cv::MORPH_CLOSE, kernel);

    // Thresholding
    cv::threshold(msk, msk, 128, 255, cv::THRESH_BINARY);

    return msk;
}

// Function to remove a border from a mask
cv::Mat removeBorder(const cv::Mat& msk, int width = 50) {
    assert(msk.type() == CV_8U, "msk must be a grayscale image");

    cv::Mat result = msk.clone();

    int dh = msk.rows / width;
    int dw = msk.cols / width;

    // Set the top, bottom, left, and right borders to 255
    result.rowRange(0, dh).setTo(cv::Scalar(255));
    result.rowRange(result.rows - dh, result.rows).setTo(cv::Scalar(255));
    result.colRange(0, dw).setTo(cv::Scalar(255));
    result.colRange(result.cols - dw, result.cols).setTo(cv::Scalar(255));

    return result;
}

// Function to create a blur mask with additional processing
std::tuple<cv::Mat, double, bool> blurMask(const cv::Mat& img) {
    assert(img.type() == CV_8UC3, "img_col must be a color image");

    // Perform your initial blur detection here (not provided in the Python code)

    cv::Mat msk, blurry;
    double result;

    // Invert the mask and adjust intensity levels
    cv::convertScaleAbs(255 - (255 * msk / cv::norm(msk, cv::NORM_L2)));
    cv::threshold(msk, msk, 50, 255, cv::THRESH_BINARY);
    cv::threshold(msk, msk, 127, 255, cv::THRESH_BINARY_INV);

    // Remove borders from the mask
    msk = removeBorder(msk);

    // Apply morphological operations
    msk = morphology(msk);

    // Calculate the percentage of the image that is blurry
    result = static_cast<double>(cv::countNonZero(msk)) / (255.0 * msk.size());

    std::cout << result * 100 << "% of input image is blurry" << std::endl;

    return std::make_tuple(msk, result, blurry);
}

int main() {
    // Read the image
    cv::Mat img = cv::imread("path/to/your/image.jpg");
    
    // Obtain the blur mask and its value
    auto [msk, val, blurry] = blurMask(img);

    // Display the original image and the blur mask
    cv::imshow("img", img);
    cv::imshow("msk", msk);
    cv::waitKey(0);

    return 0;
}
