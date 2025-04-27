import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from skimage import io

from image_patchify import *
from psf_generator import generate_psfs

metalens_param = {
'aperture_diameter' : 0.2e-3,
'lambda_base' : [606.0, 511.0, 462.0],
'channel_idx' : [2, 1, 0],
'theta_base' : [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0],
'focal_length' : 1.15e-3,
'refractive_index' : 1.45,
'crop_size' : 301,
'duty_filename' : 'E:/Data/DiffractiveOpticsSimulator/Metalens/1/duty.npy'
}

if __name__ == "main":
    output_dir = './results'
    psf_array = generate_psfs(metalens_param)

    for idx in range(len(psf_array)):
        cv2.imwrite(f'{output_dir}/theta_{idx}.png', psf_array[idx])