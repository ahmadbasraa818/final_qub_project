#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include </home/thatchaoskid/Documents/FloatX/src/floatx.hpp> //Path to the floatx lib using vm
//#include </mnt/c/Users/ahmad/Documents/FloatX/src/floatx.hpp> //Path to the floatx lib using wsl

using namespace std;
using namespace cv;
using namespace flx;

// Define FloatX type for convenience
constexpr int f = 5;
constexpr int l = 10;
typedef floatx<f, l> FloatX;

// Custom sqrt function for FloatX
FloatX sqrt_floatx(const FloatX& value) {
    return FloatX(sqrt(static_cast<double>(value)));
}

// Custom trigonometric functions for FloatX
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
};

// FFT function declaration
void fft(vector<MyComplex>& a, bool inverse);

Mat visualizeFFT(const vector<MyComplex>& data, int rows, int cols) {
    Mat magnitudeImage(rows, cols, CV_32F);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int index = i * cols + j;
            FloatX mag = sqrt_floatx(data[index].real * data[index].real + data[index].imag * data[index].imag);
            magnitudeImage.at<float>(i, j) = static_cast<float>(mag);
        }
    }
    magnitudeImage += Scalar::all(1); // Avoid log(0)
    log(magnitudeImage, magnitudeImage); // Apply log
    normalize(magnitudeImage, magnitudeImage, 0, 1, NORM_MINMAX); // Normalize for display
    return magnitudeImage;
}

Mat processBlock(const Mat& block) {
    vector<MyComplex> blockData;
    for (int y = 0; y < block.rows; ++y) {
        for (int x = 0; x < block.cols; ++x) {
            blockData.push_back(MyComplex(block.at<uchar>(y, x), 0));
        }
    }

    // Visualize original block
    imshow("Original Block", block);
    waitKey(0); // Wait for key press to move on

    fft(blockData, false);

    // Visualize and show FFT of the block
    Mat fftImage = visualizeFFT(blockData, block.rows, block.cols);
    imshow("FFT Block", fftImage);
    waitKey(0); // Wait for key press to move on

    return block.clone();
}

// Blur detection function
bool isBlurry(vector<MyComplex> &freqData) {
    flx::floatx<f, l> sumMagnitude = 0.0;
    for (const auto& c : freqData) {
        sumMagnitude += sqrt_floatx(c.real * c.real + c.imag * c.imag);
    }
    flx::floatx<f, l> averageMagnitude = sumMagnitude / freqData.size();

    flx::floatx<f, l> sumHighMagnitude = 0.0;
    for (size_t i = 1; i < freqData.size(); ++i) {
        sumHighMagnitude += sqrt_floatx(freqData[i].real * freqData[i].real + freqData[i].imag * freqData[i].imag);
    }
    flx::floatx<f, l> averageHighMagnitude = sumHighMagnitude / (freqData.size() - 1);

    double thresholdRatio = 0.2;

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
void fft(vector<MyComplex>& a, bool inverse = false) {    int n = a.size();
    cout << "FFT call with vector size: " << a.size() << "\n";
    if (!isPowerOfTwo(n)) {
        int newSize = nextPowerOfTwo(n);
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


int main(int argc, char** argv) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <image_file>" << endl;
        return 1;
    }
    string image_file = argv[1];
    Mat image = imread(image_file, IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Error: Unable to load image " << image_file << endl;
        return 1;
    }

    // Corrected from 'frame' to 'image'.
    int minBlockSize = 1;
    int maxBlockSize = min(image.cols, image.rows);
    int defaultBlockSize = minBlockSize;

    cout << "Enter the block size (" << minBlockSize << " - " << maxBlockSize << "): ";
    int blockSize;
    cin >> blockSize;

    



    if (blockSize < minBlockSize || blockSize > maxBlockSize) {
        cerr << "Invalid block size. Using default size: " << defaultBlockSize << endl;
        blockSize = defaultBlockSize;
    }

    // Process each block of the image.
    for (int y = 0; y < image.rows; y += blockSize) {
        for (int x = 0; x < image.cols; x += blockSize) {
            Rect blockRect = Rect(x, y, min(blockSize, image.cols - x), min(blockSize, image.rows - y));
            Mat block = image(blockRect);

            // Process each block
            Mat processedBlock = processBlock(block);
// If you intend to show processedBlock or do further operations with it,
            // ensure those actions are performed here.
        }
    }

    // Optionally, wait for a key press to close the program or move to the next step.
    waitKey(0);
    return 0;
}
