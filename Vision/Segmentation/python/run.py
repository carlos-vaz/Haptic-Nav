import sys
#from matplotlib import pyplot as plt

if len(sys.argv) != 3:
	print("Usage: \n\tpython run.py [video file] [frame stride]")
	exit(0)

if int(sys.argv[2]) < 1:
	print("Frame stride must be 1 or greater")
	exit(0)

import numpy as np
from PIL import Image
import cv2
from DeepLabModel import DeepLabModel
from frames import VideoFrames

# Instantiate model
MODEL = DeepLabModel('Models/deeplab_model.tar.gz')
print('model loaded successfully!')

# Run the model on the video frames
vf = VideoFrames(sys.argv[1])
vf_iter = iter(vf)
fstride = int(sys.argv[2])
count = 0
for cv_img in vf_iter:
	if count % fstride != 0:
		count += 1
		continue
	pil_img = Image.fromarray(cv_img)
	resized_im, seg_map = MODEL.run(pil_img)
	labels = []
	for pix in seg_map.flatten():
		if pix not in labels:
			labels.append(pix)
	labels.sort()
	print(labels)

	#dely = np.array(resized_im).shape[0] / len(labels)
	#map_img = np.zeros(np.array(resized_im).shape)
	#i = 0
	#for color in labels:
	#	map_img[i:i+dely][10:30] = int(float(color*255)/33)
	#	print(int(float(color*255)/33))
	#	i += dely
	#map_img_rgb = cv2.cvtColor(map_img, cv2.COLOR_GRAY2BGR)
	#cv2.imshow('ads', map_img)
	#print(map_img)
	#map_img = map_img.astype(int)
	#print(map_img)	
	#cv2.imshow('dhgs', map_img)
	#cv2.waitKey(0)


	seg_map_img = seg_map.astype(float) * 255 / 33
	seg_map_img = np.uint8(seg_map_img)
	seg_map_rgb = cv2.cvtColor(seg_map_img, cv2.COLOR_GRAY2BGR)
	combined_img = np.concatenate((np.array(resized_im), seg_map_rgb), axis=1)
	cv2.imshow('Segmentation', combined_img)
	cv2.waitKey(0)
	resized_im.close()
	count += 1

