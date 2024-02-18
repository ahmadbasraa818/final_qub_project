from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Define the extension module
extensions = [
    Extension("blur_detector",
              sources=["blur_detector.pyx"],
              include_dirs=[numpy.get_include(), '/usr/include/opencv4', '/usr/include/python3.8', '/usr/local/lib/python3.8/dist-packages/numpy/core/include', '/usr/local/include'],
              libraries=["opencv_core", "opencv_imgproc", "opencv_imgcodecs", "opencv_highgui", "python3.8"],
              language="c++")  # Set language to C++
]

# Setup configuration
setup(
    ext_modules = cythonize(extensions)
)
