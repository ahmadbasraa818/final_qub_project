To run this code you need the follow 
> OpenCV

Put all images you want tested in folder "ImagesToTest"

Code to run the script
> python3 main.py --display --mask  ImagesToTest/imagewantedtested.fileextention

How to stop the code 
>Ctrl + C = Windows
>Ctrl + Z = Mac/Linux

BEFORE RUNNING CHECKS
=======================
sudo apt update && upgrade
sudo apt install python3 python3-pip ipython3
pip install opencv-python
pip install numpy==1.24.3
pip install matplotlib
sudo apt install libgl1-mesa-glx
pip install easygui
pip install scikit-image
sudo apt install libopencv-dev



TO RUN THE CODE
================
1. First compile the C++ file - g++ -g NewFFT.cpp -o NewFFT `pkg-config --cflags --libs opencv4`
2.Run the python main - python3 main.py