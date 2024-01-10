#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <complex> 
#include <vector>
#include <filesystem>
using namespace std;
namespace fs = std::filesystem;

const float pi = M_PI;  // Value of pi
const float tau = 2*M_PI;  // Value of 2*pi
complex<float> j(0, 1);  // Imaginary unit

// Function for the Cooley-Tukey FFT algorithm
void fft(vector<complex<float>>& a, bool invert, int limit = 0) {
    int n;
    if (limit == 0) {
        n = a.size();
    } else {
        n = limit;
    }

    // Bit-reversal permutation
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            swap(a[i], a[j]);
        }
    }

    complex<float> u;
    complex<float> v;

    // FFT algorithm
    for (int len = 2; len <= n; len <<= 1) {
        float ang = 2 * pi / len * (invert ? -1 : 1);
        complex<float> wlen(cos(ang), sin(ang));
        for (int i = 0; i < n; i += len) {
            int len_o2 = len/2;
            complex<float> w(1);
            for (int j = 0; j < len_o2; j++) {
                u = a[i + j];
                v = a[i + j + len_o2] * w;
                a[i + j] = u + v;
                a[i + j + len_o2] = u - v;
                w *= wlen;
            }
        }
    }

    // Scaling for inverse FFT
    if (invert) {
        for (int i = 0; i < n; i++) {
            a[i] /= n;
        }
    }
}

// Function for the Bluestein FFT algorithm
void fftblue(vector<complex<float>>& a, vector<complex<float>>& b, bool invert) {
    int n = a.size();

    // Bit-reversal permutation for two arrays
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            swap(a[i], a[j]);
            swap(b[i], b[j]);
        }
    }

    complex<float> u;
    complex<float> v;

    // FFT algorithm for two arrays
    for (int len = 2; len <= n; len <<= 1) {
        float ang = 2 * pi / len * (invert ? -1 : 1);
        complex<float> wlen(cos(ang), sin(ang));
        for (int i = 0; i < n; i += len) {
            int len_o2 = len/2;
            complex<float> w(1);
            for (int j = 0; j < len_o2; j++) {
                u = a[i + j];
                v = a[i + j + len_o2] * w;
                a[i + j] = u + v;
                a[i + j + len_o2] = u - v;

                u = b[i + j];
                v = b[i + j + len_o2] * w;
                b[i + j] = u + v;
                b[i + j + len_o2] = u - v;

                w *= wlen;
            }
        }
    }

    // Pointwise multiplication of the two arrays
    for (int i = 0; i < n; ++i) {
        a[i] *= b[i];
    }

    // Scaling for inverse FFT
    if (invert) {
        for (int i = 0; i < n; i++) {
            a[i] /= n;
        }
    }
}

// Class for Bluestein FFT implementation
class Bluestein {
public:
    int n;
    vector<complex<float>> dfft;

    // Constructor
    Bluestein(vector<float>& signal) {
        n = signal.size();
        int l = pow(2, ceil(log2(2 * n + 1)));
        float nInv = 1.0/n;
        complex<float> comp;
        float idx = 0.0;
        float onef = 1.0;

        vector<complex<float>> U_l(l);
        vector<complex<float>> V_l(l+1);
        vector<complex<float>> V_star(n);

        // Initializing vectors for Bluestein FFT
        for (int i = 0; i < n; i++) {
            comp = exp(j*pi*(idx*idx)*nInv);
            V_star[i] = onef/comp;
            U_l[i] = signal[i]/comp;
            V_l[i] = comp;
            V_l[l-i] = comp;
            idx+=1.0;
        }

        // Performing Bluestein FFT
        fftblue(U_l, V_l, false);
        fft(U_l, true);

        // Pointwise multiplication with V_star
        for (int i = 0; i < n; i++) {
            dfft.push_back(U_l[i]*V_star[i]);
        }
    }

    // Getter function for the result coefficients
    vector<complex<float>> getFourCoeff() {
        return dfft;
    } 
};

int main(int argc, char *argv[]) {
    int precision;

    // Check if precision level is provided as a command-line argument
    if (argc == 2) {
        precision = stoi(argv[1]);
    } else {
        // Ask the user for the level of precision
        cout << "Enter the level of precision (1 - 10): ";
        cin >> precision;

        // Validate the precision value
        if (precision < 1 || precision > 10) {
            cerr << "Error: Precision must be in the range 1 to 10." << endl;
            return 1;
        }
    }

    // Set precision for floating-point output
    cout << fixed << setprecision(precision);

    vector<string> imagenames;
    string folderPath = "ImagesToTest"; // Replace with the actual path to your folder

    // Check if the folder exists
    if (!fs::is_directory(folderPath)) {
        cerr << "Error: The specified folder does not exist." << endl;
        return 1;
    }

    // Read image names from the folder
    for (const auto& entry : fs::directory_iterator(folderPath)) {
        if (entry.is_regular_file()) {
            imagenames.push_back(entry.path().filename().string());
        }
    }

    // Sample blur information as a fraction of image blur (replace with actual values if available)
    vector<float> blurValues = {0.105, 0.208, 0.302, 0.053, 0.157};

    vector<float> s = {1, 2, 3, 4, 5};
    Bluestein b2(s);
    vector<complex<float>> rst2 = b2.getFourCoeff();

    // Printing the result coefficients with imagenames and complement blur information in percentage format
    for (int i = 0; i < rst2.size(); i++) {
        // Convert the fraction to percentage and multiply by 100
        float blurPercentage = blurValues[i] * 100.0;
        
        // Calculate the complement (100% - blur percentage)
        float complementPercentage = 100.0 - blurPercentage;
        
        cout << "(" << imagenames[i] << "," << complementPercentage << "%," << rst2[i].real() << "," << rst2[i].imag() << ") ";
    }

    cout << endl;

    return 0;
}

