from unittest.mock import patch, call
import unittest
from main import run_fft_analysis  # Adjust this import according to your script's structure

class TestSubprocessCall(unittest.TestCase):
    print('Running test')
    @patch('subprocess.run')
    def test_run_fft_analysis_subprocess_call(self, mock_run):
        print('Image Path Found')
        image_paths = ['/home/thatchaoskid/Documents/final_qub_project/ImagesToTest/test6.jpeg', '/home/thatchaoskid/Documents/final_qub_project/ImagesToTest/test7.png']
        print('Running subprocess Call')
        run_fft_analysis(image_paths)
        expected_calls = [call(['./NewFFT', '/home/thatchaoskid/Documents/final_qub_project/ImagesToTest/test6.jpeg'], check=True, text=True),
                          call(['./NewFFT', '/home/thatchaoskid/Documents/final_qub_project/ImagesToTest/test7.png'], check=True, text=True)]
        mock_run.assert_has_calls(expected_calls, any_order=True)

