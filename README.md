# Haptic-Nav
Navigation through touch designed for the blind

### [`c++/`](https://github.com/fullprocess/Haptic-Nav/tree/master/c++)
This is where the main haptic navigation app lives. We use Tensorflow's C API (since the C++ API is only buildable by Bazel, which makes it less practical for embedded platforms).  

### [`python/`](https://github.com/fullprocess/Haptic-Nav/tree/master/python)
(See `python/README.md` for details on Deeplab and Cityscapes) An initial testing ground for experimenting with the Deeplab graph. Not important to the final project. 

### [`Models/`](https://github.com/fullprocess/Haptic-Nav/tree/master/Models)
This directory contains a Deeplabv3 frozen graph pre-trained on Cityscapes. A frozen graph is a graph + a checkpoint that have been processed by [`freeze_graph.py`](https://github.com/tensorflow/tensorflow/blob/9849fde5e7b4da4b630ffbc517fad68b2b811c0c/tensorflow/python/tools/freeze_graph.py) so that the trainable weights from the `model.ckpt.data` file are transfered to the `model.ckpt.meta` (graph structure) file as constants. 
