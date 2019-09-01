import sys

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
MODEL = DeepLabModel('../Models/deeplab_model.tar.gz')
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
	b, g, r = pil_img.split()
	pil_img = Image.merge("RGB", (r, g, b))
	resized_im, seg_map = MODEL.run(pil_img)
	labels = []
	for pix in seg_map.flatten():
		if pix not in labels:
			labels.append(pix)
	labels.sort()
	print(labels)

	seg_map_img = seg_map.astype(float) * 255 / 33
	seg_map_img = np.uint8(seg_map_img)
	seg_map_rgb = cv2.cvtColor(seg_map_img, cv2.COLOR_GRAY2BGR)
	combined_img = np.concatenate((np.array(resized_im), seg_map_rgb), axis=1)
	cv2.imshow('Segmentation', combined_img)
	cv2.waitKey(0)
	resized_im.close()
	count += 1

