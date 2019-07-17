## Semantic Segmentation with Deeplabv3 pre-trained on the Cityscapes dataset

### Background
* *Deeplabv3* is a deep neural network architecture for semantic segmentation. This model uses MobileNetv2 as its main feature-extraction network. The model, pre-trained on the Cityscapes dataset, can be downloaded [here] (http://download.tensorflow.org/models/deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz). 
* The *Cityscapes* [dataset] (https://www.cityscapes-dataset.com) consists of 5000 human-segmented images of urban city street scenes from different cities in Germany. 

### Dependencies
This code uses Python 2.7, as well as the following python packages:
* PIL
* OpenCV
* tensorflow
* numpy
* six
* matplotlib

Use PIP to install any missing packages. 

### Running 
Simply move an .mp4 video file into this directory and name it 'footage.mp4'. Then run `python runm.py`

