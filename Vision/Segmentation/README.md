## Segmentation with Deeplab
(See `python/README.md` for details on Deeplab and Cityscapes)

## `c/`
The programs here are actually C++, but they use tensorflow's C API, since the C++ API is only buildable by Bazel, which makes it less practical for embedded platforms. The goal of the programs here is to compile into an object file that can be linked into the haptic navigation application so that the latter can input RGB images and get segmentation map outputs. 

## `python/`
An initial testing ground for experimenting with the Deeplab graph. Not important to the final project. 

## `Models/`
This directory contains a Deeplabv3 frozen graph pre-trained on Cityscapes. A frozen graph is a graph + a checkpoint that have been processed by [`freeze_graph.py`](https://github.com/tensorflow/tensorflow/blob/9849fde5e7b4da4b630ffbc517fad68b2b811c0c/tensorflow/python/tools/freeze_graph.py) so that the trainable weights from the `model.ckpt.data` are transfered to the `model.ckpt.meta` (graph structure) file as constants. 
