import numpy as np
from main import create_mask_from_fft
import unittest

class TestImageProcessing(unittest.TestCase):
    def test_create_mask_from_fft_low_frequency(self):
        # Simulate an FFT result that predominantly has low frequencies
        fft_data = np.ones((10, 10)) + 1j * np.ones((10, 10))
        mask = create_mask_from_fft(fft_data)
        # Expect most of the mask to be zeros since it's mostly low frequencies
        self.assertTrue(np.count_nonzero(mask) < 20)

    def test_create_mask_from_fft_high_frequency(self):
        # Simulate an FFT result with high frequencies
        fft_data = np.random.rand(10, 10) + 1j * np.random.rand(10, 10)
        mask = create_mask_from_fft(fft_data)
        # Expect a significant number of ones in the mask due to high frequencies
        self.assertTrue(np.count_nonzero(mask) > 20)

