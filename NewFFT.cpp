#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <floatx.hpp>

using namespace std;
using namespace cv;


using MyComplex = floatx::floatx<23>; // This is where the precision would be

// FFT function declaration
void fft(vector<MyComplex> &a, bool inverse);

// Blur detection function
bool isBlurry(vector<MyComplex> &freqData) {
    // Calculate average magnitude of frequency components
    MyComplex sumMagnitude = 0.0;
    for (const auto& c : freqData) {
        sumMagnitude += sqrt(c.real() * c.real() + c.imag() * c.imag());
    }
    MyComplex averageMagnitude = sumMagnitude / freqData.size();

    // Calculate average magnitude of high-frequency components (excluding DC component)
    MyComplex sumHighMagnitude = 0.0;
    for (size_t i = 1; i < freqData.size(); ++i) {
        sumHighMagnitude += sqrt(freqData[i].real() * freqData[i].real() + freqData[i].imag() * freqData[i].imag());
    }
    MyComplex averageHighMagnitude = sumHighMagnitude / (freqData.size() - 1);

    // Set a threshold ratio (you may need to adjust this value)
    double thresholdRatio = 0.2;

    // Check if the ratio of average high-frequency magnitude to average magnitude is below the threshold
    return (averageHighMagnitude / averageMagnitude < thresholdRatio);
}

// FFT function definition
void fft(vector<MyComplex> &a, bool inverse) {
    int n = a.size();
    if (n <= 1) return;

    // Rearrange the vector
    vector<MyComplex> a0(n / 2), a1(n / 2);
    for (int i = 0, j = 0; i < n; i += 2, j++) {
        a0[j] = a[i];
        a1[j] = a[i + 1];
    }

    // Recursive FFT for even and odd elements
    fft(a0, inverse);
    fft(a1, inverse);

    double ang = 2 * M_PI / n * (inverse ? -1 : 1);
    MyComplex w(1, 0), wn(cos(ang), sin(ang));
    for (int k = 0; k < n / 2; k++) {
        MyComplex t = w * a1[k];
        a[k] = a0[k] + t;
        a[k + n / 2] = a0[k] - t;
        if (inverse) {
            a[k] /= 2;
            a[k + n / 2] /= 2;
        }
        w *= wn;
    }
}

int main(int argc, char **argv) {
    // Check if the number of command line arguments is correct
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <image_file>\n", argv[0]);
        return 1;
    }
    string image_file = argv[1]; 
    cout << "Processing " << image_file << std::endl;
    int block_size;
    std::cout << "Enter the block size: ";
    std::cin >> block_size;

    // Load the image
    Mat frame = imread(image_file, IMREAD_GRAYSCALE);

    // Check if the image is loaded successfully
    if (frame.empty()) {
        cerr << "Error: Unable to load image: " << image_file << endl;
        return 1;
    }

    int cx = frame.cols/2;
    int cy = frame.rows/2;

    // Convert image to vector of Complex numbers
    vector<MyComplex> image_data(frame.cols * frame.rows);
    for (int y = 0; y < frame.rows; ++y) {
        for (int x = 0; x < frame.cols; ++x) {
            image_data[y * frame.cols + x] = MyComplex(frame.at<uchar>(y, x), 0);
        }
    }

    // Perform IFFT
    fft(image_data, true);

    // Scale the inverse FFT output
    for (int i = 0; i < image_data.size(); ++i) {
        image_data[i] /= image_data.size();
    }

    // Detect blur
    bool blurry = isBlurry(image_data);

    cout << "Blurry: " << (blurry ? "Yes" : "No") << endl;

    // Perform IFFT (not sure if this is intended)
    fft(image_data, true);

    // Convert Complex vector back to image
    Mat result(frame.size(), CV_8UC1);
    for (int y = 0; y < frame.rows; ++y) {
        for (int x = 0; x < frame.cols; ++x) {
            // Take real part of complex number and cast to uchar
            result.at<uchar>(y, x) = static_cast<uchar>(image_data[y * frame.cols + x].real());
        }
    }

    // Display results
    imshow("Input", frame);
    imshow("Result", result);
    waitKey();

    return 0;
}
