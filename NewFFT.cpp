#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <filesystem>
#include </home/thatchaoskid/Documents/FloatX/src/floatx.hpp>

using namespace std;
using namespace cv;
using namespace flx;
namespace fs = std::filesystem;

const double PI = acos(-1);
constexpr int f = 8;  // Number of bits for the exponent
constexpr int l = 12; // Number of bits for the significand (fraction)

// Single precision = (f = 8 bits) + (l = 23 bits) - Typical IEEE 754 single precision
// Half precision = (f = 5 bits) + (l = 10 bits) - Reduced precision, useful for graphics and machine learning
// Double precision = (f = 15 bits) + (l = 112 bits) - High precision for scientific computing (not directly supported in standard C++)
// Bfloat16 = (f = 8 bits) + (l = 7 bits) - Balances wide range with precision, popular in machine learning (requires hardware or library support)

typedef floatx<f, l> FloatX;

FloatX sqrt_floatx(const FloatX& value) {
    return FloatX(sqrt(static_cast<double>(value)));
}

FloatX cos_floatx(const FloatX& value) {
    return FloatX(cos(static_cast<double>(value)));
}

FloatX sin_floatx(const FloatX& value) {
    return FloatX(sin(static_cast<double>(value)));
}

struct MyComplex {
    FloatX real, imag;
    MyComplex(FloatX r = 0.0, FloatX i = 0.0) : real(r), imag(i) {}

    MyComplex operator+(const MyComplex& other) const {
        return MyComplex(real + other.real, imag + other.imag);
    }

    MyComplex operator-(const MyComplex& other) const {
        return MyComplex(real - other.real, imag - other.imag);
    }

    MyComplex operator*(const MyComplex& other) const {
        return MyComplex(real * other.real - imag * other.imag, real * other.imag + imag * other.real);
    }
    FloatX abs() const {
        return FloatX(sqrt(static_cast<double>(real * real + imag * imag)));
    }
};

// Declaration of functions used in the program. Definitions should follow.
void fft(vector<MyComplex>& a, bool invert); // Performs the Fast Fourier Transform on a vector of MyComplex
void saveFFTResults(const vector<vector<MyComplex>>& fftData, const string& filePath); // Saves the FFT results to a file
void transpose(vector<vector<MyComplex>>& data); // Transposes a 2D vector of MyComplex
void fft2D(vector<vector<MyComplex>>& data, bool invert); // Performs 2D FFT on a matrix of MyComplex
double calculateBlurriness(const vector<vector<MyComplex>>& freqDomain); // Calculates the blurriness of an image based on its frequency domain representation
void displayFrequencyMagnitude(const vector<vector<MyComplex>>& freqDomain); // Displays the magnitude of the frequencies in the frequency domain representation
void processSingleImage(const string& inputPath); // Processes a single image for blurriness analysis
bool isPowerOfTwo(int n); // Checks if a number is a power of two
int nextPowerOfTwo(int n); // Finds the next power of two greater than or equal to n
void displayProgress(int current, int total); // Displays a progress bar
void shiftDFT(Mat& fImage); // Shifts the zero-frequency component to the center of the spectrum

size_t SafeIndex(size_t index, size_t size) {
    if (index < size) {
        return index;
    } else {
        std::cerr << "Index out of bounds. Index: " << index << ", Size: " << size << std::endl;
        // Handle the error: exit or throw an exception
        exit(EXIT_FAILURE);
    }
}

void shiftDFT(Mat& fImage) {
    int cx = fImage.cols / 2;
    int cy = fImage.rows / 2;

    Mat q0(fImage, Rect(0, 0, cx, cy));   // Top-Left
    Mat q1(fImage, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(fImage, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(fImage, Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

int progress_line_number = 100;
void displayProgress(int current, int total) {
    const int barWidth = 70;
    float progress = static_cast<float>(current) / total;
    int pos = barWidth * progress;

    // Move up to the previous line of the progress bar
    std::cout << "\033[" << progress_line_number << ";0H";

    // If you have reached a new line (current is 0), increase the line number for the next call
    if (current == 0) {
        ++progress_line_number;
    }

    std::cout << "[";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
}


void fft(vector<MyComplex>& a, bool inverse = false) {
    int n = a.size();
    cout << "FFT call with vector size: " << n << "\n";

    if (!isPowerOfTwo(n)) {
        std::cerr << "Warning: Input size is not a power of two. Resizing to the nearest power of two.\n";
        int newSize = nextPowerOfTwo(n);
        a.resize(newSize, MyComplex()); // Pad with zeros
        n = newSize;
    }

    if (n <= 1) {
        cout << "Completed FFT for vector size: " << n << " (base case)\n";
        return;
    }

    vector<MyComplex> a_even(n / 2), a_odd(n / 2);
    for (int i = 0; i < n / 2; i++) {
        a_even[i] = a[2 * i];
        a_odd[i] = a[2 * i + 1];
    }

    fft(a_even, inverse);
    fft(a_odd, inverse);

    FloatX angle = 2 * M_PI / FloatX(n) * (inverse ? -1 : 1);
    MyComplex w(1), wn(cos_floatx(angle), sin_floatx(angle));
    for (int k = 0; k < n / 2; k++) {
        MyComplex t = w * a_odd[k];
        a[k] = a_even[k] + t;
        a[k + n / 2] = a_even[k] - t;
        if (inverse) {
            a[k].real = a[k].real / 2;
            a[k].imag = a[k].imag / 2;
            a[k + n / 2].real = a[k + n / 2].real / 2;
            a[k + n / 2].imag = a[k + n / 2].imag / 2;
        }
        w = w * wn;
    }
}
bool isPowerOfTwo(int n) {
    return (n & (n - 1)) == 0;
}

int nextPowerOfTwo(int n) {
    if (n < 1) return 1;
    int power = 1;
    while (power < n) power <<= 1;
    return power;
}

void saveFFTResults(const std::vector<std::vector<MyComplex>>& fftData, const std::string& filePath) {
    std::ofstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing FFT results." << std::endl;
        return;
    }

    for (const auto& row : fftData) {
        for (const auto& val : row) {
            file << val.real << "," << val.imag << " ";
        }
        file << "\n";
    }
    file.close();
}

void transpose(std::vector<std::vector<MyComplex>>& data) {
    int n = data.size();
    int m = data[0].size();
    std::vector<std::vector<MyComplex>> result(m, std::vector<MyComplex>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            result[j][i] = data[i][j];
        }
    }
    data = std::move(result);
}

void fft2D(std::vector<std::vector<MyComplex>>& data, bool invert) {
    int n = data.size(); // Number of rows, assuming square image for simplicity
    int totalOperations = 2 * n; // Twice because we process rows first, then columns
    int completedOperations = 0;

    // Process each row with FFT
    for (auto& row : data) {
        fft(row, invert);
        completedOperations++;
        displayProgress(completedOperations, totalOperations); // Update progress after each row
    }

    transpose(data);

    // Process each column (now row after transpose) with FFT
    for (auto& row : data) {
        fft(row, invert);
        completedOperations++;
        displayProgress(completedOperations, totalOperations); // Update progress after each column
    }

    // Ensure progress is marked complete at the end
    displayProgress(totalOperations, totalOperations);
    transpose(data);

    if (invert) {
        int n = data.size();
        int m = data[0].size();
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                data[i][j] = MyComplex(data[i][j].real / (n * m), data[i][j].imag / (n * m));
            }
        }
    }
}


double calculateBlurriness(const std::vector<std::vector<MyComplex>>& freqDomain) {
    FloatX totalEnergy = 0.0;
    FloatX highFreqEnergy = 0.0;
    int cutoff = freqDomain.size() / 5; // Example threshold for high frequencies

    for (int y = 0; y < freqDomain.size(); ++y) {
        for (int x = 0; x < freqDomain[0].size(); ++x) {
            // Use the abs() method from MyComplex for magnitude
            FloatX magnitude = freqDomain[y][x].abs();
            totalEnergy += magnitude;
            if (x > cutoff && y > cutoff) {
                highFreqEnergy += magnitude;
            }
        }
    }

    FloatX ratio = highFreqEnergy / totalEnergy;
    return ratio; // Lower ratio indicates a blurrier image
}


void displayFrequencyMagnitude(const std::vector<std::vector<MyComplex>>& freqDomain) {
    int height = freqDomain.size();
    int width = freqDomain[0].size();
    Mat magnitudeImage = Mat::zeros(height, width, CV_32F);
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Use the abs() method from MyComplex to calculate magnitude
            float magnitude = freqDomain[y][x].abs();
            magnitudeImage.at<float>(y, x) = magnitude;
        }
    }

    magnitudeImage += Scalar::all(1); // Avoid log(0)
    log(magnitudeImage, magnitudeImage);

    // Shift the zero-frequency component to the center
    shiftDFT(magnitudeImage);

    // Normalize the magnitude to a range [0, 1] for display purposes
    normalize(magnitudeImage, magnitudeImage, 0, 1, NORM_MINMAX);

    // Display the dynamic range for diagnostic purposes
    double minVal, maxVal;
    minMaxLoc(magnitudeImage, &minVal, &maxVal);
    cout << "Dynamic Range - MinVal: " << minVal << ", MaxVal: " << maxVal << endl;

    // Resize the image for better viewing if it's too large
    double maxDimension = 800; // Maximum size of the window for either width or height
    double scale = min(maxDimension / magnitudeImage.cols, maxDimension / magnitudeImage.rows);
    Mat resizedMagnitudeImage;
    resize(magnitudeImage, resizedMagnitudeImage, Size(), scale, scale, INTER_LINEAR);

    // Display the resized frequency magnitude image
    imshow("Frequency Magnitude", resizedMagnitudeImage);
    waitKey(0);
}


void processSingleImage(const std::string& inputPath) {
    // Construct the expected CSV file path
    std::string fftResultsFilePath = inputPath + "_fft_results.csv";

    // Check if the CSV file exists and delete it if it does
    if (fs::exists(fftResultsFilePath)) {
        fs::remove(fftResultsFilePath);
        std::cout << "Existing FFT results file removed: " << fftResultsFilePath << std::endl;
    }

    Mat img = imread(inputPath, IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error loading image: " << inputPath << std::endl;
        return;
    }

    imshow("Original Image", img);
    waitKey(0);

    int width = img.cols;
    int height = img.rows;

    std::vector<std::vector<MyComplex>> imageData(height, std::vector<MyComplex>(width));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            imageData[y][x] = MyComplex(img.at<uchar>(y, x), 0);
        }
    }

    fft2D(imageData, false); // Perform FFT
    saveFFTResults(imageData, fftResultsFilePath); // Save FFT results
    
    FloatX blurriness = calculateBlurriness(imageData);
    std::cout << "Blurriness: " << blurriness << std::endl;

    displayFrequencyMagnitude(imageData); // Display FFT magnitude
}

#ifndef TESTING
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <ImagePath1> <ImagePath2> ..." << std::endl;
        return -1;
    }

    for (int i = 1; i < argc; ++i) {
        std::cout << "Processing: " << argv[i] << std::endl;
        processSingleImage(argv[i]);
    }

    return 0;
}
#endif