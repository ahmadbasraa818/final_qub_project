To run this code you need the follow 
> OpenCV

Put all images you want tested in folder "ImagesToTest"

Code to run the script
> python3 main.py 

How to stop the code 
>Ctrl + C = Windows
>Ctrl + Z = Mac/Linux

BEFORE RUNNING CHECKS
=======================
1. 'sudo apt update && upgrade'
2. 'sudo apt install python3 python3-pip ipython3'
3. 'pip install opencv-python'
4. 'pip install numpy==1.24.3'
5. 'pip install matplotlib'
6. 'sudo apt install libgl1-mesa-glx'
7. 'pip install easygui'
8. 'pip install scikit-image'
9. 'sudo apt install libopencv-dev'

TO RUN THE CODE
================
1. First compile the C++ file - g++ -g NewFFT.cpp -o NewFFT `pkg-config --cflags --libs opencv4`
2.Run the python main - python3 main.py

HOW TO RUN TESTS
=================
PYTHON TESTS
=================
1. Make sure you are in the root directory of the project
2. Run the following command to test GUI Interaction - python3 -m unittest discover Tests -p GUIInteractionTest.py
3. Run the following command to test the Image Proceesing - python3 -m unittest discover Tests -p ImageProcessingTest.py
4. Make sure the .csv file for image6 and image7 are in the Images to test
5. If they are not there run the program to generate them
6. Run the following command to test the SubProcees Calls - python3 -m unittest discover Tests -p SubProcessCallTest.py

C++ TESTS
==============
1. Make sure you are in the test directory
2. Run the following command to compile the tests - g++ -o C++Tests C++Tests.cpp -std=c++17 `pkg-config --cflags --libs opencv4`
3. Run the following command to run the test suite - ./C++Tests 