cmake_minimum_required (VERSION 3.1.0)
project (Camera_Output)
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

include_directories("$ENV{LIBREALSENSE_DIR}/include")
link_directories("$ENV{LIBREALSENSE_DIR}/build")

set(SOURCES ../src/test.cpp)

add_executable(test ${SOURCES})
target_link_libraries(test librealsense2.dylib ${OpenCV_LIBS})


