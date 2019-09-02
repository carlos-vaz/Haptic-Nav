cmake_minimum_required (VERSION 3.1.0)
project (Haptic_Navigation)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

find_package(OpenCV REQUIRED)

include_directories("$ENV{LIBREALSENSE_DIR}/include")
link_directories("$ENV{LIBREALSENSE_DIR}/build")

set(SOURCES deeplab_app.cpp tiny_deeplab_api.cpp)

add_executable(deeplab_app ${SOURCES})
target_link_libraries(deeplab_app librealsense2.dylib ${OpenCV_LIBS} libtensorflow.dylib)

