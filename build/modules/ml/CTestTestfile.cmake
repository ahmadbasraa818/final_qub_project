# CMake generated Testfile for 
# Source directory: /home/thatchaoskid/Desktop/final_qub_project/opencv/modules/ml
# Build directory: /home/thatchaoskid/Desktop/final_qub_project/build/modules/ml
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_ml "/home/thatchaoskid/Desktop/final_qub_project/build/bin/opencv_test_ml" "--gtest_output=xml:opencv_test_ml.xml")
set_tests_properties(opencv_test_ml PROPERTIES  LABELS "Main;opencv_ml;Accuracy" WORKING_DIRECTORY "/home/thatchaoskid/Desktop/final_qub_project/build/test-reports/accuracy" _BACKTRACE_TRIPLES "/home/thatchaoskid/Desktop/final_qub_project/opencv/cmake/OpenCVUtils.cmake;1795;add_test;/home/thatchaoskid/Desktop/final_qub_project/opencv/cmake/OpenCVModule.cmake;1375;ocv_add_test_from_target;/home/thatchaoskid/Desktop/final_qub_project/opencv/cmake/OpenCVModule.cmake;1133;ocv_add_accuracy_tests;/home/thatchaoskid/Desktop/final_qub_project/opencv/modules/ml/CMakeLists.txt;2;ocv_define_module;/home/thatchaoskid/Desktop/final_qub_project/opencv/modules/ml/CMakeLists.txt;0;")
