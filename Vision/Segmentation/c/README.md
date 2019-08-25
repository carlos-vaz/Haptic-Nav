## 1: Installing the Tensorflow C API
Follow the instructions at <https://www.tensorflow.org/install/lang_c>. Verify that `tensorflow/c/c_api.h` is inside your global include directory (on macOS: `/usr/local/include/tensorflow/c/c_api.h`), and that the tensorflow C library is in your global lib directory (on macOS: `/usr/local/lib/libtensorflow.dylib`)


## 2: Running
```
make
./test
```

## 3: Interpreting the Deeplab output tensor
The label map can be found [here](https://github.com/tensorflow/models/issues/6991). 
