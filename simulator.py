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
    # read images
    img = cv2.imread('./data/div_000005.png')  # Replace with your image path
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, [512, 512])

    # Parameters
    patch_size = [256] * 2
    stride = patch_size[0] // 2
    padding_size = 32   # independent padding size
    blending_pad = patch_size[0] // 2   # blending area size
    patches, masks, positions = split_image_into_patches(img, patch_size, stride, padding_size, blending_pad)
    print(f"Number of patches: {len(patches)}")
    print(f"First patch position (relative to original image): {positions[0]}")
    print(f"Patch size: {patches[0].shape}, Mask size: {masks[0].shape}")
    # Visualization
    visualize_patches(img, patches, masks, positions, patch_size, stride, padding_size)



    # # generate psfs
    # psf_array = generate_psfs(metalens_param)
    # for idx in range(len(psf_array)):
    #     cv2.imwrite(f'{output_dir}/theta_{idx}.png', psf_array[idx])

    # # split image into patches
