#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <iostream>
#include <vector>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

using namespace std;
namespace fs = std::filesystem;

std::vector<cv::Mat> getMasks(const cv::Mat& img) {
    // Convert the image to grayscale
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // Threshold the grayscale image
    cv::threshold(gray, gray, 128, 255, cv::THRESH_BINARY);

    // Find contours in the thresholded image
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(gray, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Create masks for each contour
    std::vector<cv::Mat> masks;
    for (const auto& contour : contours) {
        cv::Mat mask = cv::Mat::zeros(img.size(), CV_8U);
        cv::drawContours(mask, std::vector<std::vector<cv::Point>>{contour}, 0, cv::Scalar(255), cv::FILLED);
        masks.push_back(mask);
    }

    return masks;
}

// Function for morphological operations on a mask
cv::Mat morphology(const cv::Mat& msk) {
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

cv::Mat removeBorder(const cv::Mat& msk, int width = 50) {
    cv::Mat result = msk.clone();

    // Define border dimensions
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
    cv::Mat msk = img.clone();
    double val = 0.0;
    bool blurry = false;

    // Invert the mask and adjust intensity levels
    cv::Mat invertedMask;
    cv::convertScaleAbs(255 - (255 * msk / cv::norm(msk, cv::NORM_INF)), invertedMask);
    cv::threshold(invertedMask, invertedMask, 50, 255, cv::THRESH_BINARY);
    cv::threshold(invertedMask, invertedMask, 127, 255, cv::THRESH_BINARY_INV);

    // Remove borders from the mask
    cv::Mat borderRemovedMask = removeBorder(invertedMask);

    // Apply morphological operations
    cv::Mat processedMask = morphology(borderRemovedMask);

    // Calculate the percentage of the image that is blurry
    double result = cv::sum(processedMask)[0] / (255.0 * processedMask.size().area());
    std::cout << static_cast<int>(100 * result) << "% of input image is not blurry" << std::endl;

    return std::make_tuple(processedMask, result, blurry);
}

int main() {
    // Create a socket
    int server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket < 0) {
        std::cerr << "Error creating socket\n";
        return 1;
    }

    // Bind the socket to localhost and a port
    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(12345); // Port number
    if (bind(server_socket, (struct sockaddr *)&address, sizeof(address)) < 0) {
        std::cerr << "Error binding socket\n";
        return 1;
    }

    // Listen for incoming connections
    if (listen(server_socket, 5) < 0) {
        std::cerr << "Error listening on socket\n";
        return 1;
    }

    while (true) {
        // Accept incoming connection
        int client_socket = accept(server_socket, NULL, NULL);
        if (client_socket < 0) {
            std::cerr << "Error accepting connection\n";
            return 1;
        }

        // Receive image data from Python client
        cv::Mat img;
        int img_size;
        recv(client_socket, &img_size, sizeof(int), 0);
        std::vector<uchar> buffer(img_size);
        recv(client_socket, buffer.data(), img_size, 0);
        img = cv::imdecode(buffer, cv::IMREAD_COLOR);

        // Process the received image
        auto [msk, val, blurry] = blurMask(img);

        // Serialize the mask
        std::vector<uchar> serialized_mask;
        cv::imencode(".jpg", msk, serialized_mask);

        // Send the serialized mask back to the Python client
        int mask_size = serialized_mask.size();
        send(client_socket, &mask_size, sizeof(int), 0);
        send(client_socket, serialized_mask.data(), mask_size, 0);

        // Close the client socket
        close(client_socket);
    }

    // Close the server socket (this will not be reached in this code)
    close(server_socket);

    return 0;
}
