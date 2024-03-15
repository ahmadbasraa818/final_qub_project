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

================For the focus mask2 ======================

g++ -o FocusMask2 FocusMask2.cpp -I/usr/local/include/opencv4 -L/usr/local/includez -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs



python3 main.py


===============For the focus mask =======================

g++ FocusMask2.cpp -o FocusMask2

python3 main.py

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

g++ -o NewFFT NewFFT.cpp -I/usr/local/include/opencv4 -I/usr/local/include/FloatX/src -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs


g++ -fsanitize=address -g NewFFT.cpp -o NewFFT -I/usr/local/include/opencv4 -I/usr/local/include/FloatX/src -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs

