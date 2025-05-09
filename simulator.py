import torch
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from skimage import io

from image_patchify import *
from psf_generator import generate_psfs

# metalens_param = {
# 'aperture_diameter' : 0.2e-3,
# 'lambda_base' : [630.0, 540.0, 460.0],
# 'channel_idx' : [2, 1, 0],
# 'theta_base' : [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0],
# 'prop_length' : 1.15e-3,
# 'refractive_index' : 1.45,
# 'crop_size' : 201,
# 'duty_filename' : './data/duty.npy',
# 'psf_pixel_size': 350e-9,
# 'image_pixel_size': 1.75e-6,
# #
# 'patch_size': 128,
# 'padding_size': 16,
# 'image_size': 512,
# 'visualze': False
# }

metalens_param = {
'aperture_diameter' : 0.7e-3,
'lambda_base' : [630.0, 540.0, 460.0],
'channel_idx' : [2, 1, 0],
'theta_base' : [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0],
'prop_length' : 3e-3,
'refractive_index' : 1,
'crop_size' : 401,
'duty_filename' : './data/duty_3mm.npy',
'psf_pixel_size': 350e-9,
'image_pixel_size': 2.5e-6,
#
'patch_size': 200,
'padding_size': 32,
'image_size': 800,
'visualze': False
}

def generate_dot_background(image_size=(512, 512), array_size=(10, 10), dot_radius=5, save_path=None):
    """
    Generate a black background with white dots.

    Parameters:
    - image_size: (width, height) of the image in pixels
    - array_size: (columns, rows) number of dots
    - dot_radius: radius of each white dot in pixels
    - save_path: if provided, save the image to this path

    Returns:
    - img: generated image as a numpy array
    """
    width, height = image_size
    cols, rows = array_size

    # Create a black image
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Compute spacing
    x_spacing = width // (cols + 1)
    y_spacing = height // (rows + 1)

    # Draw white dots
    for i in range(1, cols + 1):
        for j in range(1, rows + 1):
            center = (i * x_spacing, j * y_spacing)
            cv2.circle(img, center, dot_radius, (255, 255, 255), thickness=-1)

    if save_path:
        cv2.imwrite(save_path, img)

    return img

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
    for dist in distances:
        dist_um = dist * pixel_size
        theta_angle = np.arctan2(dist_um, flocal_length)
        theta_angle = np.degrees(theta_angle)
        theta_angles.append(theta_angle)

    return distances, angles, theta_angles

def map_to_sorted_unique_indices(arr):
    """
    Map each element in arr to its position in the sorted unique list.

    Args:
        arr (np.ndarray): Input 1D numpy array (np.float32).

    Returns:
        unique_sorted (np.ndarray): Sorted unique values.
        mapped_indices (np.ndarray): For each element in arr, its index in unique_sorted.
    """
    unique_vals = np.unique(arr)
    unique_sorted = np.sort(unique_vals)

    # Create a mapping: value -> index
    val_to_index = {val: idx for idx, val in enumerate(unique_sorted)}

    # Now map the original array
    mapped_indices = np.array([val_to_index[val] for val in arr], dtype=np.int32)

    return unique_sorted, mapped_indices

def resize_and_rotate_psf(psf, psf_pixel_size, image_pixel_size, angle_degrees):
    """
    Resize and rotate PSF while keeping its center fixed.

    Args:
        psf (np.ndarray): Input PSF (2D numpy array).
        psf_pixel_size (float): Pixel size of the PSF.
        image_pixel_size (float): Pixel size of the target image.
        angle_degrees (float): Rotation angle in degrees (counter-clockwise).

    Returns:
        np.ndarray: Resized and rotated PSF.
    """
    # Step 1: Rotate PSF first while keeping the center fixed
    center = (psf.shape[1] / 2.0, psf.shape[0] / 2.0)
    rotation_matrix = cv2.getRotationMatrix2D(center, -angle_degrees, 1.0)
    rotated_psf = cv2.warpAffine(
        psf, 
        rotation_matrix, 
        (psf.shape[1], psf.shape[0]), 
        flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_REFLECT_101
    )
    # Step 2: Resize PSF based on pixel size difference
    scale = psf_pixel_size / image_pixel_size
    new_size = (int(rotated_psf.shape[1] * scale), int(rotated_psf.shape[0] * scale))
    resized_psf = cv2.resize(rotated_psf, new_size, interpolation=cv2.INTER_CUBIC)

    normalized_psf = resized_psf.astype(np.float32)
    for c in range(3):
        normalized_psf[:, :, c] = normalized_psf[:, :, c] / np.sum(normalized_psf[:, :, c])
    return normalized_psf

if __name__ == "__main__":
    print('Start simulator ...')
    output_dir = './results'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'psf'), exist_ok=True)
    generate_dot_background(image_size=(metalens_param['image_size'], metalens_param['image_size']), 
                            array_size=(9, 9), dot_radius=3, save_path='./data/div_0.png')
    image_names = ['div_0', 'div_000000', 'div_000002', 'div_000005', 'div_000006', 'div_000015', 'div_000044']

    for img_idx in range(len(image_names)):
        # read images
        imagename = f'./data/{image_names[img_idx]}.png'
        print(f'Process {imagename} ...')
        img = cv2.imread(imagename)  # Replace with your image path
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_size = [metalens_param['image_size'], metalens_param['image_size']]
        img = cv2.resize(img, image_size)

        # Parameters
        patch_size = [metalens_param['patch_size']] * 2
        stride = patch_size[0] // 2
        padding_size = metalens_param['padding_size']   # independent padding size
        blending_pad = patch_size[0] // 2   # blending area size
        patches, masks, positions = split_image_into_patches(img, patch_size, stride, padding_size, blending_pad)
        if metalens_param['visualze'] == True:
            print(f"Number of patches: {len(patches)}")
            print(f"First patch position (relative to original image): {positions[0]}")
            print(f"Patch size: {patches[0].shape}, Mask size: {masks[0].shape}")

        # Compute patch positions and psf angle and rotations
        if img_idx == 0:
            distances, angles, theta_angles = convert_position_to_angle_rotation(positions, image_size, 
                                                                                metalens_param['image_pixel_size'], 
                                                                                metalens_param['prop_length']/metalens_param['refractive_index'])
            print('Patch distance to center (px):', distances)
            print('Rotation for psf (px):', angles)
            print('Theta for psf simulation:', theta_angles)
            theta_angles_sorted, theta_indices = map_to_sorted_unique_indices(theta_angles)
            print('Theta induces:', theta_indices)
            print('Theta values sorted:', theta_angles_sorted)
            # generate psfs
            metalens_param['theta_base'] = theta_angles_sorted
            psf_array = generate_psfs(metalens_param)
            for idx in range(len(psf_array)):
                cv2.imwrite(f'{output_dir}/psf/theta_{idx}.png', psf_array[idx])
        
        # applying psfs
        filtered_patches = []
        for idx in range(len(patches)):
            patch = patches[idx]
            angle = angles[idx]
            psf_idx = theta_indices[idx]
            psf = psf_array[psf_idx]
            # rotate and resize psf
            psf_aligned = resize_and_rotate_psf(psf, metalens_param['psf_pixel_size'], 
                                            metalens_param['image_pixel_size'], 
                                            angle)
            # apply image filter
            # print('Pos: ', positions[idx], 'psf idx: ', psf_idx)
            filtered_patch = np.zeros_like(patch)
            for c in range(3):  # For each channel (R, G, B)
                filtered_patch[:, :, c] = cv2.filter2D(patch[:, :, c], -1, psf_aligned[:, :, 2 - c])
            filtered_patches.append(filtered_patch)

        merged_image = merge_patches(filtered_patches, masks, positions, image_size, patch_size, padding_size)
        cv2.imwrite(f'{output_dir}/{image_names[img_idx]}.png', cv2.cvtColor(merged_image, cv2.COLOR_RGB2BGR))

        if metalens_param['visualze'] == True:
            # Visualization
            visualize_patches(img, patches, masks, positions, patch_size, stride, padding_size)
            # (Optional) show a blending mask separately
            plt.figure(figsize=(4, 4))
            plt.imshow(merged_image)
            plt.title('Merged image')
            plt.show()

