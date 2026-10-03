#define TESTING
#include "/home/thatchaoskid/Documents/final_qub_project/NewFFT.cpp"

#include <iostream>
#include <cassert>
#include <cmath>
#include <limits>

// Utility function for comparing floating-point numbers
bool nearlyEqual(double a, double b, double epsilon = 1e-5) {
    return std::fabs(a - b) < epsilon;
}

void testIsPowerOfTwo() {
    // Test edge cases and general functionality
    assert(isPowerOfTwo(1) && "isPowerOfTwo failed for n = 1");
    assert(isPowerOfTwo(2) && "isPowerOfTwo failed for n = 2");
    assert(!isPowerOfTwo(3) && "isPowerOfTwo failed for n = 3");
    assert(isPowerOfTwo(1024) && "isPowerOfTwo failed for n = 1024");
    assert(!isPowerOfTwo(1023) && "isPowerOfTwo failed for n = 1023");
    std::cout << "isPowerOfTwo tests passed." << std::endl;
}

void testNextPowerOfTwo() {
    // Test edge cases and general functionality
    assert(nextPowerOfTwo(0) == 1 && "nextPowerOfTwo failed for n = 0");
    assert(nextPowerOfTwo(1) == 1 && "nextPowerOfTwo failed for n = 1");
    assert(nextPowerOfTwo(2) == 2 && "nextPowerOfTwo failed for n = 2");
    assert(nextPowerOfTwo(3) == 4 && "nextPowerOfTwo failed for n = 3");
    assert(nextPowerOfTwo(1023) == 1024 && "nextPowerOfTwo failed for n = 1023");
    assert(nextPowerOfTwo(1025) == 2048 && "nextPowerOfTwo failed for n = 1025");
    std::cout << "nextPowerOfTwo tests passed." << std::endl;
}

void testMyComplexOperations() {
    // Test addition
    MyComplex a(1.0, 2.0), b(3.0, 4.0);
    MyComplex result = a + b;
    assert(nearlyEqual(result.real, 4.0) && nearlyEqual(result.imag, 6.0) && "MyComplex addition failed");

    // Test subtraction
    result = a - b;
    assert(nearlyEqual(result.real, -2.0) && nearlyEqual(result.imag, -2.0) && "MyComplex subtraction failed");

    // Test multiplication
    result = a * b;
    assert(nearlyEqual(result.real, -5.0) && nearlyEqual(result.imag, 10.0) && "MyComplex multiplication failed");

    // Test abs (magnitude)
    double magnitude = result.abs();
    assert(nearlyEqual(magnitude, std::sqrt(125.0)) && "MyComplex abs (magnitude) failed");

    std::cout << "MyComplex operation tests passed." << std::endl;
}

// Example test for FFT functionality
void testFFT() {
    std::vector<MyComplex> data = {{0, 0}, {1, 0}, {0, 0}, {1, 0}};
    fft(data, false);  // Perform FFT
    for (const auto& val : data) {
        std::cout << "FFT result: (" << val.real << ", " << val.imag << ")\n";
    }
    // Here you would assert on expected values. This is an illustrative example;
    // exact assertions would depend on the expected outcome of your FFT operation.
    std::cout << "FFT tests passed." << std::endl;
}

int main() {
    std::cout << "Starting detailed tests...\n";
    testIsPowerOfTwo();
    testNextPowerOfTwo();
    testMyComplexOperations();
    testFFT();  // Example placeholder for more detailed FFT testing
    std::cout << "All detailed tests completed successfully.\n";
    return 0;
}
