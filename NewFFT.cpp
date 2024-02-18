#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char **argv) {
    if (argc <= 2) {
        fprintf(stderr, "Usage: %s <BLOCK> <image_file>\n", argv[0]); //Each blcok is set to 60 pizels inclease = more precise
        return 1;
    }
    int block_size = atoi(argv[1]); // Convert the second argument to an integer (BLOCK size)
    string image_file = argv[2];

    cout << "Processing " << image_file << std::endl;
    Mat frame = imread(image_file, IMREAD_GRAYSCALE);

    int cx = frame.cols/2;
    int cy = frame.rows/2;

    // Go float
    Mat fImage;
    frame.convertTo(fImage, CV_32F);

    // FFT
    cout << "Direct transform...\n";
    Mat fourierTransform;
    dft(fImage, fourierTransform, DFT_SCALE|DFT_COMPLEX_OUTPUT);

    //center low frequencies in the middle
    //by shuffling the quadrants.
    Mat q0(fourierTransform, Rect(0, 0, cx, cy));       // Top-Left - Create a ROI per quadrant
    Mat q1(fourierTransform, Rect(cx, 0, cx, cy));      // Top-Right
    Mat q2(fourierTransform, Rect(0, cy, cx, cy));      // Bottom-Left
    Mat q3(fourierTransform, Rect(cx, cy, cx, cy));     // Bottom-Right

    Mat tmp;                                            // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                                     // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);

    // Block the low frequencies
    fourierTransform(Rect(cx-block_size, cy-block_size, 2*block_size, 2*block_size)).setTo(0);

    //shuffle the quadrants to their original position
    Mat orgFFT;
    fourierTransform.copyTo(orgFFT);
    Mat p0(orgFFT, Rect(0, 0, cx, cy));       // Top-Left - Create a ROI per quadrant
    Mat p1(orgFFT, Rect(cx, 0, cx, cy));      // Top-Right
    Mat p2(orgFFT, Rect(0, cy, cx, cy));      // Bottom-Left
    Mat p3(orgFFT, Rect(cx, cy, cx, cy));     // Bottom-Right

    p0.copyTo(tmp);
    p3.copyTo(p0);
    tmp.copyTo(p3);

    p1.copyTo(tmp);                                     // swap quadrant (Top-Right with Bottom-Left)
    p2.copyTo(p1);
    tmp.copyTo(p2);

    // IFFT
    cout << "Inverse transform...\n";
    Mat invFFT;
    Mat logFFT;
    double minVal,maxVal;

    dft(orgFFT, invFFT, DFT_INVERSE|DFT_REAL_OUTPUT);

    //img_fft = 20*numpy.log(numpy.abs(img_fft))
    invFFT = cv::abs(invFFT);
    cv::minMaxLoc(invFFT,&minVal,&maxVal,NULL,NULL);
    
    //check for impossible values
    if(maxVal<=0.0){
        cerr << "No information, complete black image!\n";
        return 1;
    }

    cv::log(invFFT,logFFT);
    logFFT *= 20;

    //result = numpy.mean(img_fft)
    cv::Scalar result= cv::mean(logFFT);
    cout << "Result : "<< result.val[0] << endl;

    Mat finalImage;
    logFFT.convertTo(finalImage, CV_8U);    // Back to 8-bits
    imshow("Input", frame);
    imshow("Result", finalImage);  //Higher value = sharper image  level of blur present in image 
    cv::waitKey();

    return 0;
}

    // What this does
    // 1. Includes necessary libraries and namespaces: iostream, opencv2/opencv.hpp.
    // 2. Defines a constant BLOCK which is used to set the size of the block for blocking low frequencies.
    // 3. The main function takes the input image file path as a command-line argument.
    // 4. Reads the input image in grayscale.
    // 5. Converts the image to float.
    // 6. Performs the FFT (forward DFT) on the float image.
    // 7. Shifts the zero frequency component to the center of the spectrum.
    // 8. Blocks low frequencies by setting a portion of the spectrum to zero.
    // 9. Shifts the spectrum back to its original quadrant arrangement.
    // 10. Performs the inverse FFT (inverse DFT) to obtain the spatial domain image.
    // 11. Computes the logarithm of the absolute value of the inverse FFT result and scales it by 20.
    // 12. Calculates the mean value of the logarithmic magnitude spectrum.
    // 13. Converts the logarithmic magnitude spectrum to 8-bit for visualization.
    // 14. Displays the original grayscale image and the result of the blur detection.
    // 15. Waits for a key press to close the windows and end the program.
