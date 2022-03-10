from PIL import Image
import numpy as np
import os
import random

class DataReader:
	def __init__(self, directory):
		self.image_dir = os.path.join(directory, 'images')
		self.fixation_dir = os.path.join(directory, 'fixations')
		self.images = os.listdir(self.image_dir)

		self.num_images = len(self.images)
		self.index = 0

	def num_images(self):
		return self.num_images

	def num_batches_of_size(self, batchsize):
		return int(np.ceil(float(self.num_images) / batchsize))

	def read_image(self, name):
		# Read and convert to floating point
		im = np.array(Image.open(name)).astype(np.float32)
		if im.ndim == 2: # If we read grayscale image, expand the dimensions
			im = im[:, :, np.newaxis]

		im /= 255.0 # normalize to range [0, 1]
		return im

	def shuffle(self):
		random.shuffle(self.images)

	def get_batch(self, batchsize):
		first = self.index
		last = min(self.index + batchsize, self.num_images)
		real_batchsize = last - first
		self.index = last

		images_to_read = self.images[first:last]

		img = np.empty((real_batchsize, 224, 224, 3), np.float32)
		fix = np.empty((real_batchsize, 224, 224, 1), np.float32)

		for x, image in enumerate(images_to_read):
			img_name = os.path.join(self.image_dir, image)
			fix_name = os.path.join(self.fixation_dir, image)

			img[x, :, :, :] = self.read_image(img_name)
			fix[x, :, :, :] = self.read_image(fix_name)


		if self.index == self.num_images:
			# This is the last batch
			# Reset index and shuffle list of images
			self.index = 0
			self.shuffle()

		return img, fix, images_to_read