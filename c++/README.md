## Haptic Navigation C++

### 1. Install the Tensorflow C API
Follow the instructions at <https://www.tensorflow.org/install/lang_c>. Verify that `tensorflow/c/c_api.h` is inside your global include directory (on macOS: `/usr/local/include/tensorflow/c/c_api.h`), and that the tensorflow C library is in your global lib directory (on macOS: `/usr/local/lib/libtensorflow.dylib`)

### 2. Install librealsense
[macOS](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_osx.md)  
[Ubuntu Linux](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md)  
[Windows](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_windows.md)  

### 3. Install OpenCV
On a Mac, [use Homebrew](https://www.pyimagesearch.com/2016/12/19/install-opencv-3-on-macos-with-homebrew-the-easy-way/)

### 4. Environment variables
```
$ export LIBREALSENSE_DIR=/path/to/librealsense
```

### 5. Build & run
```
$ mkdir build && cd build
$ cmake ..
$ make
$ ../bin/app
```

