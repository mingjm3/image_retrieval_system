cmake_minimum_required(VERSION 3.15)

set (CMAKE_CXX_STANDARD 17)

# Put your compiled file name in NAME
project( ${{ NAME }} )

# Put which file will be compiled
# Example: compile 'app.cpp', FILE should be 'app.cpp'
set(SOURCE src/ ${{ FILE }} )

# Opencv environment is required, set the path to your opencv environment
INCLUDE_DIRECTORIES(/usr/local/include/opencv4)

# Link your library
LINK_DIRECTORIES(/usr/local/lib)

add_executable(${PROJECT_NAME} ${SOURCE})

# FLANN is required
find_package(OpenCV REQUIRED COMPONENTS flann)

# Dependencies
TARGET_LINK_LIBRARIES(${PROJECT_NAME} libopencv_core.dylib libopencv_highgui.dylib libopencv_imgcodecs.dylib libopencv_imgproc.407.dylib opencv_features2d opencv_calib3d)