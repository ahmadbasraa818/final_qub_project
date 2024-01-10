# CMake generated Testfile for 
# Source directory: /home/thatchaoskid/Desktop/final_qub_project/opencv/modules/highgui
# Build directory: /home/thatchaoskid/Desktop/final_qub_project/build/modules/highgui
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_highgui "/home/thatchaoskid/Desktop/final_qub_project/build/bin/opencv_test_highgui" "--gtest_output=xml:opencv_test_highgui.xml")
set_tests_properties(opencv_test_highgui PROPERTIES  LABELS "Main;opencv_highgui;Accuracy" WORKING_DIRECTORY "/home/thatchaoskid/Desktop/final_qub_project/build/test-reports/accuracy" _BACKTRACE_TRIPLES "/home/thatchaoskid/Desktop/final_qub_project/opencv/cmake/OpenCVUtils.cmake;1795;add_test;/home/thatchaoskid/Desktop/final_qub_project/opencv/cmake/OpenCVModule.cmake;1375;ocv_add_test_from_target;/home/thatchaoskid/Desktop/final_qub_project/opencv/modules/highgui/CMakeLists.txt;309;ocv_add_accuracy_tests;/home/thatchaoskid/Desktop/final_qub_project/opencv/modules/highgui/CMakeLists.txt;0;")
