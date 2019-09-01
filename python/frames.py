import cv2

class VideoFrames:
	def __init__(self, videofile):
		self.vid = cv2.VideoCapture(videofile)

	def __iter__(self):
		self.success, self.image = self.vid.read()
		return self

	def next(self):
		if self.success:
			ret_img = self.image
			self.success, self.image = self.vid.read()
			return ret_img
		else:
			raise StopIteration

