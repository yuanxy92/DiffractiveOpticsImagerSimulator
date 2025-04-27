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

def convert_position_to_angle_rotation(positions, image_size, pixel_size, flocal_length):
    """
    Compute distance and angle from each patch center to the image center.
    
    Args:
        positions (list of (x, y)): List of patch center coordinates.
        image_size (tuple): (width, height) of the original image.
        return_in_degrees (bool): If True, angles are returned in degrees. Else, in radians.

    Returns:
        distances (list of floats): Euclidean distances.
        angles (list of floats): Angles (0 = positive x-axis).
    """
    img_w, img_h = image_size
    center_x = img_w / 2
    center_y = img_h / 2
    distances = []
    angles = []

    for (x, y) in positions:
        dx = x - center_x
        dy = y - center_y
        distance = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)  # angle relative to x-axis

        angle = np.degrees(angle)
        distances.append(distance)
        angles.append(angle)
    
    theta_angles = []
    for dist in distance:
        dist_um = dist * pixel_size
        theta_angle = np.arctan2(dist_um, flocal_length)
        theta_angle = np.degrees(theta_angle)
        theta_angles.append(theta_angle)

    return distances, angles, theta_angles

if __name__ == "__main__":
    print('Start simulator ...')
    output_dir = './results'
    # read images
    img = cv2.imread('./data/div_000005.png')  # Replace with your image path
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_size = [512, 512]
    img = cv2.resize(img, image_size)

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

    distances, angles, theta_angles = convert_position_to_angle_rotation(positions, image_size, 1.75e-6, 1.15e-3)
    print(distances)
    print(angles)
    print(theta_angles)


    # # generate psfs
    # psf_array = generate_psfs(metalens_param)
    # for idx in range(len(psf_array)):
    #     cv2.imwrite(f'{output_dir}/theta_{idx}.png', psf_array[idx])

    # # split image into patches
