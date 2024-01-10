To run this code you need the follow 
> OpenCV

Put all images you want tested in folder "ImagesToTest"

Code to run the script
> python3 main.py --display --mask  ImagesToTest/imagewantedtested.fileextention

How to stop the code 
>Ctrl + C = Windows
>Ctrl + Z = Mac/Linux

g++ -o your_cpp_executable FocusMask.cpp -I /usr/local/include/opencv4 -L/usr/local/include/opencv4/lib -lopencv_core -lopencv_highgui -lopencv_imgproc

1 = Not precisie
10 = Very precisise
