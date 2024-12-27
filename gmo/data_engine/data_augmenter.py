from __future__ import print_function
import math
import numpy
import cv2
import torch
from torchvision import datasets
from torchvision import transforms as tf
from albumentations import *
from albumentations.pytorch import ToTensorV2

# custom transformations from albumentations library
class AlbumentationTransformations():
  def __init__(self, means, stdevs):
    self.means = numpy.array(means)
    self.stdevs = numpy.array(stdevs)
    patch_size = 32#28
    self.album_transforms = Compose([
      HorizontalFlip(p = 0.5),
      ShiftScaleRotate(p=0.5),
      CoarseDropout (max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16, fill_value=self.means, mask_fill_value = None),
	    # Cutout(num_holes=1, max_h_size=8, max_w_size=8, fill_value=self.means*255.0, p=0.75),
      Normalize(mean=self.means, std=self.stdevs),
      ToTensorV2()
    ])
        
  def __call__(self, img):
      img = numpy.array(img)
      img = self.album_transforms(image=img)['image']
      return img


def augment_data(means, stdevs):
  transformations = tf.Compose([
    tf.RandomCrop(32, padding=4),
    tf.RandomHorizontalFlip(),
    tf.ToTensor(),
    tf.Normalize(means, stdevs)
  ])
  return transformations