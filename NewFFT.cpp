#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// Complex number structure
struct MyComplex {
    double real, imag;
    MyComplex() : real(0.0), imag(0.0) {}
    MyComplex(int r, int i) : real(r), imag(i) {}

    MyComplex operator+(const MyComplex& other) const {
        return MyComplex(real + other.real, imag + other.imag);
    }

    MyComplex operator-(const MyComplex& other) const {
        return MyComplex(real - other.real, imag - other.imag);
    }

    MyComplex operator*(const MyComplex& other) const {
        return MyComplex(real * other.real - imag * other.imag,
                         real * other.imag + imag * other.real);
    }
};

// FFT function declaration
void fft(vector<MyComplex> &a, bool inverse);

// Blur detection function
bool isBlurry(vector<MyComplex> &freqData) {
    // Calculate average magnitude of frequency components
    double sumMagnitude = 0.0;
    for (const auto& c : freqData) {
        sumMagnitude += sqrt(c.real * c.real + c.imag * c.imag);
    }
    double averageMagnitude = sumMagnitude / freqData.size();

    // Calculate average magnitude of high-frequency components (excluding DC component)
    double sumHighMagnitude = 0.0;
    for (size_t i = 1; i < freqData.size(); ++i) {
        sumHighMagnitude += sqrt(freqData[i].real * freqData[i].real + freqData[i].imag * freqData[i].imag);
    }
    double averageHighMagnitude = sumHighMagnitude / (freqData.size() - 1);

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
            a[k].real /= 2;
            a[k].imag /= 2;
            a[k + n / 2].real /= 2;
            a[k + n / 2].imag /= 2;
        }
        w = w * wn;
    }
}

int main(int argc, char **argv) {
    if (argc <= 2) {
        fprintf(stderr, "Usage: %s <BLOCK> <image_file>\n", argv[0]);
        return 1;
    }

    int block_size = atoi(argv[1]);
    string image_file = argv[2];

    cout << "Processing " << image_file << std::endl;
    
    Mat frame = imread(image_file, IMREAD_GRAYSCALE);

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
        image_data[i].real /= image_data.size();
        image_data[i].imag /= image_data.size();
    }

    // Detect blur
    bool blurry = isBlurry(image_data);

    cout << "Blurry: " << (blurry ? "Yes" : "No") << endl;

    // Perform IFFT
    fft(image_data, true);

    // Convert Complex vector back to image
    Mat result(frame.size(), CV_8UC1);
    for (int y = 0; y < frame.rows; ++y) {
        for (int x = 0; x < frame.cols; ++x) {
            // Take real part of complex number and cast to uchar
            result.at<uchar>(y, x) = static_cast<uchar>(image_data[y * frame.cols + x].real);
        }
    }

    // Display results
    imshow("Input", frame);
    imshow("Result", result);
    waitKey();

    return 0;
}
