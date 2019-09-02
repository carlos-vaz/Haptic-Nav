# Haptic-Nav
Navigation through touch for the blind

## Step by step

#### 1. Install the Tensorflow C API
We use Tensorflow's C API (since the C++ API is only buildable by Bazel, which makes it less practical for embedded platforms). Follow the instructions at <https://www.tensorflow.org/install/lang_c>. Verify that `tensorflow/c/c_api.h` is inside your global include directory (on macOS: `/usr/local/include/tensorflow/c/c_api.h`), and that the tensorflow C library is in your global lib directory (on macOS: `/usr/local/lib/libtensorflow.dylib`)

#### 2. Install librealsense
[macOS](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_osx.md)  
[Ubuntu Linux](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md)  
[Windows](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_windows.md)  

#### 3. Install OpenCV
On a Mac, [use Homebrew](https://www.pyimagesearch.com/2016/12/19/install-opencv-3-on-macos-with-homebrew-the-easy-way/)

#### 4. Environment variables
Tell Cmake the location of librealsense:
```
$ export LIBREALSENSE_DIR=/path/to/cloned/librealsense/directory
```

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

