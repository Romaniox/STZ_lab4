cmake_minimum_required(VERSION 3.16)
project(lab4)

set(CMAKE_CXX_STANDARD 14)

add_executable(lab4 main.cpp custom_DFT.cpp custom_DFT.h custom_FFT.cpp custom_FFT.h)

set(OpenCV_DIR "D:\\OpenCV\\mingw-build\\install")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)

target_link_libraries(lab4 ${OpenCV_LIBS})