## Semantic Segmentation with Deeplabv3 pre-trained on the Cityscapes dataset
Based on [this demo](https://colab.research.google.com/github/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb)

### Background
* *Deeplabv3* is a deep neural network architecture for semantic segmentation. This model uses MobileNetv2 as its main feature-extraction network. The model, pre-trained on the Cityscapes dataset, can be downloaded [here](http://download.tensorflow.org/models/deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz). 
* The *Cityscapes* [dataset](https://www.cityscapes-dataset.com) consists of 5000 human-segmented images of urban city street scenes from different cities in Germany. 

### Dependencies
This code uses Python 2.7, as well as the following python packages:
* PIL
* OpenCV
* tensorflow
* numpy
* matplotlib

Use PIP to install any missing packages. 

### Running 
First, move an .mp4 video file into this directory. Then run, for example, `python  run.py  my_video_file.mp4  1`, where the last argument '1' tells the program to segment every frame in the video (a '100' would mean 100th frame will get segmented). Wait for the pre-trained model to load from the Models directory, and eventually two windows will appear displaying the original first frame, and the segmentation output. Simply press any key to proceed to move on to the next frame. 

