cmake_minimum_required (VERSION 3.1.0)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

find_package(OpenCV REQUIRED)

set(SOURCES deeplab_app.cpp tiny_deeplab_api.cpp)

set(CMAKE_PATH_TO_MODELS_DIR "${Haptic_Navigation_SOURCE_DIR}/Models")
configure_file(
	"${Haptic_Navigation_SOURCE_DIR}/cmake/modelpath_CMAKE.hpp.in"
	"${CMAKE_CURRENT_SOURCE_DIR}/modelpath_CMAKE.hpp"
)

add_executable(deeplab_app ${SOURCES})
target_link_libraries(deeplab_app librealsense2.dylib ${OpenCV_LIBS} libtensorflow.dylib)

