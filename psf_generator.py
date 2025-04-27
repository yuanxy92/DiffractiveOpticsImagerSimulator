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

from waveoptics.optical_units import *
from waveoptics.metalens_param import *
params['padding_dim'] = 1000

device = torch.device('cuda:0')
duty_filename = 'E:/Data/DiffractiveOpticsSimulator/Metalens/1/duty.npy'
output_dir = './results'
# os.makedirs(output_dir, exist_ok=True)

def compute_and_crop_center_position(psf_img):
    psf_img = psf_img / np.max(psf_img)
    thresh_val, psf_img2 = cv2.threshold(psf_img, 0.5, 1, cv2.THRESH_TOZERO)
    # Get the dimensions of the image
    height, width = psf_img2.shape
    # Create a grid of coordinates
    y_indices, x_indices = np.indices((height, width))
    # Calculate the sum of all pixel values
    total_intensity = psf_img2.sum()
    # Calculate the weighted sum of coordinates
    x_weighted_sum = (x_indices * psf_img2).sum()
    y_weighted_sum = (y_indices * psf_img2).sum()
    # Calculate the weighted center (centroid)
    x_center = x_weighted_sum / total_intensity
    y_center = y_weighted_sum / total_intensity
    return x_center, y_center

def compute_and_crop_center_position_w_angle(psf_img, theta, phi, distance, pixel_size, rindex, crop_half_size):
    psf_img = psf_img / np.max(psf_img)
    theta_out = math.asin(1 / rindex * math.sin(math.radians(theta)))
    phi_out = math.asin(1 / rindex * math.sin(math.radians(phi)))
    # compute pixel shift
    pixel_shift_theta = distance * math.tan(theta_out) / pixel_size
    pixel_shift_phi = distance * math.tan(phi_out) / pixel_size
    # Calculate the weighted center (centroid)
    x_center = psf_img.shape[1] // 2 - pixel_shift_theta
    y_center = psf_img.shape[0] // 2 - pixel_shift_phi

    # crop blocks from image
    x_lf = x_center - crop_half_size
    if x_lf < 0:
        psf_img2 = np.copy(psf_img)
        cut_pos = psf_img2.shape[1] // 2
        psf_img2 = np.roll(psf_img2, [0, cut_pos])
        x_center2 = x_center + cut_pos
    else:
        psf_img2 = psf_img
        x_center2 = x_center

    xc = math.floor(x_center2)
    yc = math.floor(y_center)
    output_mat_crop = psf_img2[yc - crop_half_size:yc + crop_half_size + 1, xc - crop_half_size:xc + crop_half_size + 1]

    return x_center, y_center, output_mat_crop


def generate_psfs(metalens_param):
    # set parameter
    aperture_diameter = metalens_param["aperture_diameter"]
    lambda_base = metalens_param["lambda_base"]
    channel_idx = metalens_param["channel_idx"]
    theta_base = metalens_param["theta_base"]
    refractive_index = metalens_param["refractive_index"]
    crop_size = metalens_param["crop_size"]
    crop_size_half = crop_size // 2
    prop_dist = []
    prop_dist.append(metalens_param["focal_length"])

    # load duty of metalens 
    duty = np.load(duty_filename)
    duty = np.clip(duty, 0.3, 0.82)
    padding_values = ((0, 0), (0, 0))
    padding_size_duty_ = (2285 - duty.shape[0]) // 2
    padding_size_duty = ((padding_size_duty_, padding_size_duty_), (padding_size_duty_, padding_size_duty_))
    duty = np.pad(duty, padding_size_duty, 'constant', constant_values=padding_values)

    # set aperture
    params['pixels_aperture'] = aperture_diameter / params['pitch']
    aperture_mask_ = ((x_mesh ** 2 + y_mesh ** 2) < (params['pixels_aperture'] * params['Lx'] / 2.0) ** 2)
    if aperture_diameter > 0.38e-3:
        aperture_mask_inner_ = ((x_mesh ** 2 + y_mesh ** 2) < (0.38e-3 / params['pitch'] * params['Lx'] / 2.0) ** 2)
        aperture_mask_ = (aperture_mask_ + aperture_mask_inner_ * 1) / 2
    # padding
    padding_size = ((params['padding_dim'], params['padding_dim']), (params['padding_dim'], params['padding_dim']))
    padding_values = ((0, 0), (0, 0))
    aperture_mask = np.pad(aperture_mask_, padding_size, 'constant', constant_values=padding_values)

    psf_array = []
    for dist in prop_dist:
        for theta in theta_base:
            psf_array_ = np.zeros((crop_size, crop_size, 3), np.uint8)
            for lambda_idx in range(len(lambda_base)):
                lambda_ = lambda_base[lambda_idx]
                params['lam0'] = lambda_ * params['nanometers']
                prop = AngSpecProp(params['whole_dim'] + 2 * params['padding_dim'], params['pitch'], dist, params['lam0'], refractive_index)
                prop.to(device=device)

                image_amp = aperture_mask
                init_phase = generate_phase_wavefront(params['whole_dim'], params['whole_dim'], params['lam0'], theta, phi, params['pitch'])
                phase = phase_from_duty_and_lambda(duty, params)
                # image_amp = np.ones(phase.shape, dtype=np.float32)
                image_phase = init_phase + phase
                image_phase = np.pad(image_phase, padding_size, 'constant', constant_values=padding_values)
                image_complex = image_amp * np.cos(image_phase) + 1j * image_amp * np.sin(image_phase)
                image_complex_tensor = torch.from_numpy(image_complex).cuda()
                x = prop(image_complex_tensor)
                output_int = x.abs() ** 2
                input = image_complex_tensor.abs().cpu().detach().numpy()
                output_mat = output_int.abs().cpu().detach().numpy()[::-1, ::-1]
                output_mat = output_mat / np.max(output_mat) * 255
                output_mat = output_mat.astype(np.uint8)
                # compute center
                xc, yc, output_mat_crop = compute_and_crop_center_position_w_angle(output_mat, theta, phi, dist, params['pitch'], refractive_index, crop_size_half)
                output_rgb = np.zeros((output_mat_crop.shape[0], output_mat_crop.shape[1], 3), dtype=np.uint8)
                output_rgb[:, :, channel_idx[lambda_idx]] = output_mat_crop * 255
                # output_rgb_whole = np.zeros((output_mat.shape[0], output_mat.shape[1], 3), dtype=np.uint8)
                # output_rgb_whole[:, :, channel_idx[lambda_idx]] = output_mat
                psf_array_ = psf_array_ + output_rgb
            psf_array.append(psf_array_)

    return psf_array

# for idx in range(len(psf_array)):
#     cv2.imwrite(f'{output_dir}/theta_{theta_base[idx]:02.0f}.png', psf_array[idx])