# Haptic-Nav
Navigation through touch for the blind

## Step by step

#### 1. Install the Tensorflow C Library
We use Tensorflow's C API (since the C++ API is only buildable by Bazel, which makes it less practical for embedded platforms). Follow the instructions at <https://www.tensorflow.org/install/lang_c>, but for macOS or Linux, it boils down to this:

```
$ wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.14.0.tar.gz
$ sudo tar -C /usr/local -xzf libtensorflow-cpu-linux-x86_64-1.14.0.tar.gz
```
Now the header file c_api.hpp and the library libtensorflow.so.1 (or .dylib for macOS) are in the default search paths for the compiler. So if you write a test.cpp file, you can compile and run it:
```
$ g++ test.cpp -o test -ltensorflow
$ ./test              # Works in macOS, but in Linux you get:
./test: error while loading shared libraries: libtensorflow.so.1: cannot open shared object file: No such file or directory
```
On Linux, you have to update your dynamic linker's cache so it can find libtensorflow at load time
```
$ sudo ldconfig       # Only for Linux
```

Verify that `tensorflow/c/c_api.h` is inside your global include directory (on macOS or Linux: `/usr/local/include/tensorflow/c/c_api.h`), and that the tensorflow C library is in your global lib directory (macOS/Linux: `/usr/local/lib/libtensorflow.dylib` (or `.so` for Linux)). For Linux, also update the dynamic linker run-time bindings with `ldconfig` 


#### 2. Install librealsense
[macOS](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_osx.md)  
[Ubuntu Linux](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md)  
[Windows](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_windows.md)  

#### 3. Install OpenCV
On a Mac, [use Homebrew](https://www.pyimagesearch.com/2016/12/19/install-opencv-3-on-macos-with-homebrew-the-easy-way/)
On Linux, use apt-get:
```
sudo apt-get install libopencv-dev
```

#### 4. Environment variables
Tell Cmake the location of librealsense:
```
$ export LIBREALSENSE_DIR=/path/to/cloned/librealsense/directory
```ssss
#### 5. Build & run
```
$ mkdir build && cd build
$ cmake ..
$ make
$ ../bin/app
```

## Directory organization

### [`python/`](https://github.com/fullprocess/Haptic-Nav/tree/master/python)
(See `python/README.md` for details on Deeplab and Cityscapes) An initial testing ground for experimenting with the Deeplab graph. Not important to the final project. 

### [`Models/`](https://github.com/fullprocess/Haptic-Nav/tree/master/Models)
This directory contains a Deeplabv3 frozen graph pre-trained on Cityscapes. A frozen graph is a graph + a checkpoint that have been processed by [`freeze_graph.py`](https://github.com/tensorflow/tensorflow/blob/9849fde5e7b4da4b630ffbc517fad68b2b811c0c/tensorflow/python/tools/freeze_graph.py) so that the trainable weights from the `model.ckpt.data` file are transfered to the `model.ckpt.meta` (graph structure) file as constants. 

