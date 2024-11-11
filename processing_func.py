import numpy as np

#------------------ FUNCTION PROCESSING --------------------#
def grayscaling(image_process):
  grayscale_image = np.dot(image_process[...,:3], [0.3989, 0.5870, 0.1140]) # [0.3989, 0.5870, 0.1140]
  return grayscale_image

def black_and_white(image_process):
  grayscale_image = np.dot(image_process[...,:3], [0.3989, 0.5870, 0.1140])
  black_and_white_image = (grayscale_image > 128).astype(np.uint8) * 255
  return black_and_white_image

def negative(image_process):
  negative_image = 255 - image_process
  return negative_image

