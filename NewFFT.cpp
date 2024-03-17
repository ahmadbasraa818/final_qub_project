#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

struct MyComplex {
    double real, imag;
    MyComplex(double r = 0.0, double i = 0.0) : real(r), imag(i) {}

    MyComplex operator+(const MyComplex& other) const {
        return MyComplex(real + other.real, imag + other.imag);
    }

    MyComplex operator-(const MyComplex& other) const {
        return MyComplex(real - other.real, imag - other.imag);
    }

    MyComplex operator*(const MyComplex& other) const {
        return MyComplex(real * other.real - imag * other.imag, real * other.imag + imag * other.real);
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

bool isPowerOfTwo(int n) {
    return (n & (n - 1)) == 0;
}

int nextPowerOfTwo(int n) {
    if (n < 1) return 1;
    int power = 1;
    while (power < n) power <<= 1;
    return power;
}


// FFT function definition
void fft(vector<MyComplex>& a, bool inverse = false) {
    cout << "FFT call with vector size: " << a.size() << "\n";
    
    int n = a.size();
    if (!isPowerOfTwo(n)) {
        int newSize = nextPowerOfTwo(n);
        cout << "Resizing vector to: " << newSize << "\n";
        a.resize(newSize, MyComplex());
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

    double angle = 2 * M_PI / n * (inverse ? -1 : 1);
    MyComplex w(1), wn(cos(angle), sin(angle));
    for (int k = 0; k < n / 2; k++) {
        MyComplex t = w * a_odd[k];
        a[k] = a_even[k] + t;
        a[k + n / 2] = a_even[k] - t;
        if (inverse) {
            a[k].real /= 2;
            a[k].imag /= 2;
            a[k + n / 2].real /= 2;
            a[k + n / 2].imag /= 2;
        }
        w = w * wn;
    }
    cout << "Completed FFT for vector size: " << n << "\n";
}

Mat recombineBlocks(const vector<Mat>& blocks, int rows, int cols, int blockSize) {
    Mat output = Mat::zeros(rows, cols, CV_8U); // Initialize the output image
    int index = 0;
    for (int y = 0; y < rows; y += blockSize) {
        for (int x = 0; x < cols; x += blockSize) {
            // Ensure we don't exceed the image bounds
            int actualBlockSizeY = min(blockSize, rows - y);
            int actualBlockSizeX = min(blockSize, cols - x);
            Mat block = blocks[index++](Rect(0, 0, actualBlockSizeX, actualBlockSizeY));
            block.copyTo(output(Rect(x, y, actualBlockSizeX, actualBlockSizeY)));
        }
    }
    return output;
}


Mat processBlock(const Mat& block) {
    vector<MyComplex> blockData;
    for (int y = 0; y < block.rows; ++y) {
        for (int x = 0; x < block.cols; ++x) {
            blockData.push_back(MyComplex(block.at<uchar>(y, x), 0));
        }
    }

    fft(blockData, false);
    return block.clone();
}


int main(int argc, char** argv) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <image_file>" << endl;
        return 1;
    }
    string image_file = argv[1];
    cout << "Processing " << image_file << endl;

    Mat frame = imread(image_file, IMREAD_GRAYSCALE);
    if (frame.empty()) {
        cerr << "Error: Unable to load image " << image_file << endl;
        return 1;
    }

    vector<Mat> processedBlocks;
    int minBlockSize = 1;
    int maxBlockSize = min(frame.cols, frame.rows);
    int defaultBlockSize = minBlockSize;

    cout << "Enter the block size (" << minBlockSize << " - " << maxBlockSize << "): ";
    int blockSize;
    cin >> blockSize;

    if (blockSize < minBlockSize || blockSize > maxBlockSize) {
        cerr << "Invalid block size. Using default size: " << defaultBlockSize << endl;
        blockSize = defaultBlockSize;
    }

    blockSize = nextPowerOfTwo(blockSize);

    for (int y = 0; y < frame.rows; y += blockSize) {
        for (int x = 0; x < frame.cols; x += blockSize) {
            Rect blockRect(x, y, min(blockSize, frame.cols - x), min(blockSize, frame.rows - y));
            Mat block = frame(blockRect);
            processedBlocks.push_back(processBlock(block));
        }
    }

    // Recombine processed blocks into a single output image
    Mat result = recombineBlocks(processedBlocks, frame.rows, frame.cols, blockSize);

    // Display the result
    imshow("Result", result);
    waitKey(0);

    return 0;
}