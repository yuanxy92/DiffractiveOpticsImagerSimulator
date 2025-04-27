import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.fft import fftshift, fft2, ifft2, ifftshift
from torchvision import transforms
from matplotlib import pyplot as plt
import cv2
import scipy.io as sio
import math
import fnmatch

in_path = 'D:/Code/Aurora_papers/results/in_air_20240711_780um_aperture'
out_path = 'D:/Code/Aurora_papers/results/in_air_20240711_780um_aperture_2.5um'
os.makedirs(out_path, exist_ok=True)

scale = 350e-9 / 2.5e-6

# for filename in os.listdir(dir_path):
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff']
# second iteration process all the images
for root, dirnames, filenames in os.walk(in_path):
    # Check each file extension in the image_extensions list
    for extension in image_extensions:
        # Find files that match the current extension
        for filename in fnmatch.filter(filenames, extension):
            image_path = os.path.join(root, filename)
            img = cv2.imread(image_path)
            w = round(img.shape[1] * scale)
            h = round(img.shape[0] * scale)
            img_s = cv2.resize(img, [w, h])
            cv2.imwrite(f'{out_path}/{filename}', img_s)

            

            