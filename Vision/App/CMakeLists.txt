cmake_minimum_required (VERSION 3.1.0)
project (App)
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

include_directories("$ENV{LIBREALSENSE_DIR}/include")
link_directories("$ENV{LIBREALSENSE_DIR}/build")

set(SOURCES ../src/app.cpp ../../Segmentation/c/tiny_deeplab_api.cpp)

add_executable(app ${SOURCES})
target_link_libraries(app librealsense2.dylib ${OpenCV_LIBS} libtensorflow.dylib)

