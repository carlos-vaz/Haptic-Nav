import os
import numpy as np
from PIL import Image
import tensorflow as tf

class DeepLabModel(object):
	"""Class to load deeplab model and run inference."""

	INPUT_TENSOR_NAME = 'ImageTensor:0'
	OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
	INPUT_SIZE = 513

	def __init__(self, frozengraph_path):
		"""Creates and loads pretrained deeplab model."""
		self.graph = tf.Graph()

		file_handle = file(frozengraph_path)
		graph_def = tf.GraphDef.FromString(file_handle.read())

		with self.graph.as_default():
			tf.import_graph_def(graph_def, name='')
			tensors = [n.values() for n in tf.get_default_graph().get_operations()]
			#tensors = [n for n in tf.get_default_graph().as_graph_def().node]

		self.sess = tf.Session(graph=self.graph)
		print tensors

	def run(self, image):
		"""Runs inference on a single image.

		Args:
		image: A PIL.Image object, raw input image.

		Returns:
		resized_image: RGB image resized from original input image.
		seg_map: Segmentation map of `resized_image`.
		"""
		width, height = image.size
		resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
		target_size = (int(resize_ratio * width), int(resize_ratio * height))
		resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
		print("Image size: " + str(np.asarray(resized_image).shape))
		batch_seg_map = self.sess.run(self.OUTPUT_TENSOR_NAME, feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
		seg_map = batch_seg_map[0]
		return resized_image, seg_map

