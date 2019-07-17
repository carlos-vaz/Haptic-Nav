import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib
import urllib as urllib_2
import numpy as np
from PIL import Image
import cv2
from DeepLabModel import DeepLabModel
from frames import VideoFrames


def url_to_cv_image(url):
	resp = urllib_2.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	return cv2.imdecode(image, cv2.IMREAD_COLOR)

def url_to_pil_image(url):
	f = urllib.request.urlopen(url)
	jpeg_str = f.read()
	return Image.open(BytesIO(jpeg_str))


# Download and instantiate model
MODEL_URL = 'http://download.tensorflow.org/models/deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz'
_TARBALL_NAME = 'deeplab_model.tar.gz'
model_dir = tempfile.mkdtemp()
download_path = os.path.join(model_dir, _TARBALL_NAME)
print('downloading model, this might take a while...')
urllib.request.urlretrieve(MODEL_URL, download_path)
print('download completed! loading DeepLab model...')
MODEL = DeepLabModel(download_path)
print('model loaded successfully!')


# Run the model on the video frames
vf = VideoFrames('footage.mp4')
vf_iter = iter(vf)
for cv_img in vf_iter:
	pil_img = Image.fromarray(cv_img)
	resized_im, seg_map = MODEL.run(pil_img)
	print(seg_map.shape)
	seg_map = seg_map.astype(float) * 255 / np.amax(seg_map)
	cv2.imshow('segmentation', np.uint8(seg_map))
	cv2.imshow('original', np.array(resized_im))
	cv2.waitKey(0)
	resized_im.close()


