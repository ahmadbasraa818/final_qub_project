#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <filesystem>
#include </home/thatchaoskid/Documents/FloatX/src/floatx.hpp> //Path to the floatx lib using vm

using namespace std;
using namespace cv;
using namespace flx;
namespace fs = std::filesystem;

const double PI = acos(-1);
constexpr int f = 100;
constexpr int l = 150;
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

void fft(vector<MyComplex>& a, bool invert);
void saveFFTResults(const vector<vector<MyComplex>>& fftData, const string& filePath);
void transpose(vector<vector<MyComplex>>& data);
void fft2D(vector<vector<MyComplex>>& data, bool invert);
double calculateBlurriness(const vector<vector<MyComplex>>& freqDomain);
void displayFrequencyMagnitude(const vector<vector<MyComplex>>& freqDomain);
void processSingleImage(const string& inputPath);
bool isPowerOfTwo(int n);
int nextPowerOfTwo(int n);
void displayProgress(int current, int total);

size_t SafeIndex(size_t index, size_t size) {
    if (index < size) {
        return index;
    } else {
        std::cerr << "Index out of bounds. Index: " << index << ", Size: " << size << std::endl;
        // Handle the error: exit or throw an exception
        exit(EXIT_FAILURE);
    }
}


void displayProgress(int current, int total) {
    const int barWidth = 70;
    float progress = static_cast<float>(current) / total;
    int pos = barWidth * progress;
    cout << "[";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) cout << "=";
        else if (i == pos) cout << ">";
        else cout << " ";
    }
    cout << "] " << int(progress * 100.0) << " %\r";
    cout.flush();
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

    magnitudeImage += Scalar::all(1);
    log(magnitudeImage, magnitudeImage);
    normalize(magnitudeImage, magnitudeImage, 0, 1, NORM_MINMAX);
    
    imshow("Frequency Magnitude", magnitudeImage);
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